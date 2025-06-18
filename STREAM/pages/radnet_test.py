import os
from collections import OrderedDict
import json
import pandas as pd
import joblib
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
from shapely.geometry import Polygon
#from encoder import ra_encoder
#from util import DisplayHMI
import streamlit as st # Import streamlit
import polarTransform

# --- Original Constants and Model Definitions (keep these as they are) ---
NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

# Assuming Experimental.learnable_fft_wip2 exists and SignalProcessLayer is available
# If this is not a separate file, you'll need to include its definition here.
try:
    from Experimental.learnable_fft_wip2 import SignalProcessLayer
except ImportError:
    st.error("Error: Could not import SignalProcessLayer. Make sure 'Experimental/learnable_fft_wip2.py' is in your project path and contains SignalProcessLayer.")
    # You might want to define a dummy class or handle this more gracefully for a robust app
    class SignalProcessLayer(nn.Module):
        def __init__(self, use_fft_weights=True):
            super().__init__()
            st.warning("Using a dummy SignalProcessLayer. Actual processing might fail.")
        def forward(self, x):
            # Placeholder for actual signal processing
            return x.real + 0j # Return real part as complex for split_real_imag


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Detection_Header(nn.Module):
    def __init__(self, use_bn=True,reg_layer=2,input_angle_size=0):
        super(Detection_Header, self).__init__()
        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle parameter !')
            return

        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)
         
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out

class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer,out_layer,kernel_size=(1,12),dilation=(1,16),use_bn = False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size, 
                              stride=(1, 1), padding=0,dilation=dilation, bias= (not use_bn) )
    
        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna/2)

    def forward(self,x):
        width = x.shape[-1]
        x = torch.cat([x[...,-self.padding:],x,x[...,:self.padding]],axis=3)
        x = self.conv(x)
        x = x[...,int(x.shape[-1]/2-width/2):int(x.shape[-1]/2+width/2)]

        if self.use_bn:
            x = self.bn(x)
        return x

class FPN_BackBone(nn.Module):
    def __init__(self, num_block,channels,block_expansion,mimo_layer,use_bn=True):
        super(FPN_BackBone, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # pre processing block to reorganize MIMO channels
        self.pre_enc = MIMO_PreEncoder(32,mimo_layer,
                                       kernel_size=(1,NbTxAntenna),
                                       dilation=(1,NbRxAntenna),
                                       use_bn = True)

        self.in_planes = mimo_layer

        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])
                                         
    def forward(self, x):
        x = self.pre_enc(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
     
        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample,expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out

class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()
     
        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
     
        self.conv_block4 = BasicBlock(48,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)

        self.L3  = nn.Conv2d(192, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(160, 224, kernel_size=1, stride=1,padding=0)
     
    def forward(self,features):
        T4 = features['x4'].transpose(1, 3) 
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
     
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
     
        return out


class FFTRadNet(nn.Module):
    def __init__(self,mimo_layer,channels,blocks,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FFTRadNet, self).__init__()
 
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.FPN = FPN_BackBone(num_block=blocks,channels=channels,block_expansion=4, mimo_layer = mimo_layer,use_bn = True)
        self.RA_decoder = RangeAngle_Decoder()
     
        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=channels[3]*4,reg_layer=regression_layer)

        if(self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))

    def forward(self,x):
                   
        out = {'Detection':[],'Segmentation':[]}

        x = x.permute(0, 3, 1, 2)
        features= self.FPN(x)
        RA = self.RA_decoder(features)

        if(self.detection_head):
            out['Detection'] = self.detection_header(RA)

        if(self.segmentation_head):
            Y =  F.interpolate(RA, (256, 224))
            out['Segmentation'] = self.freespace(Y)
     
        return out
def interleave_real_imag(x: torch.Tensor) -> torch.Tensor:
    """
    Interleaves the real and imaginary parts of a complex tensor along the last dimension.
    If input is [..., C], output will be [..., C*2] where C*2 contains
    [R1, I1, R2, I2, ..., Rn, In].
    """
    if not torch.is_complex(x):
        raise ValueError("Input tensor must be complex.")

    real = x.real
    imag = x.imag

    
    out = torch.stack([real, imag], dim=-1).reshape(*x.shape[:-1], -1)

   
    return out

def split_real_imag(x: torch.Tensor) -> torch.Tensor:
    """
    Splits a complex tensor into real and imaginary parts along the last dimension.
    """
    if not torch.is_complex(x):
        raise ValueError("Input tensor must be complex.")

    real = x.real 
    imag = x.imag 

    # Concatenate along the last dimension
    out = torch.cat([real, imag], dim=-1) 
    return out

# --- Streamlit Application ---

# Use st.cache_resource to load the model and signal processing layer only once
@st.cache_resource
def load_model(config_path, checkpoint_path):
    """Loads the FFTRadNet model and SignalProcessLayer."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    st.write(f'Device: {device}')

    # Load configuration
    config = json.load(open(config_path))

    # Initialize FFTRadNet model
    net = FFTRadNet(blocks = config['model']['backbone_block'],
                    mimo_layer  = config['model']['MIMO_output'],
                    channels = config['model']['channels'], 
                    regression_layer = 2, 
                    detection_head = config['model']['DetectionHead'], 
                    segmentation_head = config['model']['SegmentationHead'])
    net.to(device)
    st.write('FFTRadnet initiated.')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    old_state_dict = checkpoint['net_state_dict']

    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        if k.startswith('backbone.'):
            k = k.replace('backbone.', 'FPN.')
        if k.startswith('RAmap_header.'):
            k = k.replace('RAmap_header.', 'RA_decoder.')
        if k.startswith('FPN.preproc.'):
            k = k.replace('FPN.preproc.', 'FPN.pre_enc.')
        new_state_dict[k] = v

    try:
        net.load_state_dict(new_state_dict)
        st.write('Model loaded successfully with remapped keys!')
    except RuntimeError as e:
        st.error(f"Error after remapping: {e}")
        st.warning("Attempting to load state dict with strict=False due to error.")
        try:
            net.load_state_dict(new_state_dict, strict=False)
            st.warning("Model loaded with strict=False. Some keys might be missing or mismatched.")
        except Exception as inner_e:
            st.error(f"Failed to load model even with strict=False: {inner_e}")
            return None, None # Indicate failure

    net.eval() # Set model to evaluation mode

    # Initialize signal processing layer
    signal_process_layer = SignalProcessLayer(use_fft_weights=True).to(device)
    
    return net, signal_process_layer, device

def encoder_decode(map,threshold):
    geometry={'ranges': [512, 896, 1], 'resolution': [0.201171875, 0.2], 'size': 3}
    OUTPUT_DIM=(3, 128, 224)
    statistics={'input_mean': [-0.0026244, -0.21335, 0.018789, -1.4427, -0.37618, 1.3594, -0.22987, 0.12244, 1.7359, -0.65345, 0.37976, 5.5521, 0.77462, -1.5589, -0.72473, 1.5182, -0.37189, -0.088332, -0.16194, ...], 'input_std': [20775.3809, 23085.5, 23017.6387, 14548.6357, 32133.5547, 28838.8047, 27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133, 33552.9258, 25958.7539, 29532.623, 32646.8984, 20728.332, 23160.8828, 23069.0449, ...], 'reg_mean': [0.4048094369863972, 0.3997392847799934], 'reg_std': [0.6968599580482511, 0.6942950877813826]}

    
    range_bins,angle_bins = np.where(map[0,:,:]>=threshold)

    coordinates = []

    for range_bin,angle_bin in zip(range_bins,angle_bins):
        R = range_bin*4*geometry['resolution'][0] + map[1,range_bin,angle_bin] * statistics['reg_std'][0] + statistics['reg_mean'][0]
        A = (angle_bin-OUTPUT_DIM[2]/2)*4*geometry['resolution'][1] + map[2,range_bin,angle_bin] * statistics['reg_std'][1] + statistics['reg_mean'][1]
        C = map[0,range_bin,angle_bin]
        
        coordinates.append([R,A,C])
       
    return coordinates
def RA_to_cartesian_box(data):
    L = 4
    W = 1.8
    
    boxes = []
    for i in range(len(data)):
        
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L,data[i][0],data[i][1]])
              
    return boxes

def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious
def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :8], sorted_box_predictions[i + 1:, :8])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions
    
    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_point_cloud_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))

    return final_point_cloud_predictions

camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080
def worldToImage(x,y,z):
    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0,imgpts[0][0][0]),ImageWidth-1))
    v = int(min(max(0,imgpts[0][0][1]),ImageHeight-1))
    
    return u,v

def process_predictions(model_outputs,input,image):
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    out_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0,0]
    pred_obj = encoder_decode(pred_obj,0.05)
    pred_obj = np.asarray(pred_obj)
    if(len(pred_obj)>0):
        pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)
    FFT = np.abs(input[...,:16]+input[...,16:]*1j).mean(axis=2)
    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR)

    ## Image
    for box in pred_obj:
        box = box[1:]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image = cv2.rectangle(image, (u1,v1), (u2,v2), (0, 0, 255), 3)

    RA_cartesian,_=polarTransform.convertToCartesianImage(np.moveaxis(out_seg,0,1),useMultiThreading=True,
        initialAngle=0, finalAngle=np.pi,order=0,hasColor=False)
    
    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:,256-100:256+100]
    
    RA_cartesian = np.asarray((RA_cartesian*255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian,dsize=(400,512))
    RA_cartesian=cv2.flip(RA_cartesian,flipCode=-1)

    return np.hstack((PowerSpectrum,image[:512],RA_cartesian))
    

def run_inference_and_plot(net, signal_process_layer, device, idx, adc_folder):
    """Runs inference for a given idx and generates plots."""
    st.write(f"Processing raw_adc_{idx}.npy")
    
    adc_path = os.path.join(adc_folder, f'raw_adc_{idx}.npy')
    if not os.path.exists(adc_path):
        st.error(f"ADC file not found: {adc_path}. Please check the data path and file existence.")
        return

    try:
        sample_adc = np.load(adc_path)
    except Exception as e:
        st.error(f"Error loading ADC data for index {idx}: {e}")
        return

    batch_sample_adc = torch.tensor(np.expand_dims(sample_adc, axis=0), dtype=torch.complex64)
    
    with torch.no_grad(): 
        signal_processed = signal_process_layer(batch_sample_adc.to(device))
        #signal_processed = split_real_imag(signal_processed)
        signal_processed=interleave_real_imag(signal_processed)
        prediction = net(signal_processed)

    detection_output = prediction['Detection']
    segmentation_output = prediction['Segmentation']

    
    hmi=process_predictions(model_outputs=prediction,input=sample_adc,image='/home/christophe/ComplexNet/STREAM/auto_radar.jpg')
    

    cv2.imshow('FFTRadNet',hmi)
        
    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    #sys.exit('decoded predictions not done')
    

    # --- Plot Segmentation Output ---
    st.subheader("Segmentation FFT Output")
    spatial_map = segmentation_output.squeeze().cpu().detach().numpy()
    fig1, ax1 = plt.subplots(figsize=(8, 7)) # Increased figure size for better display in Streamlit
    im1 = ax1.imshow(spatial_map, cmap='viridis') # Changed cmap to 'viridis' for better contrast
    fig1.colorbar(im1, ax=ax1, label='Intensity')
    ax1.set_title(f"Segmentation Output for idx: {idx}")
    ax1.set_xlabel("Angle (bins)")
    ax1.set_ylabel("Range (bins)")
    st.pyplot(fig1)
    plt.close(fig1) # Close the figure to free up memory

    # --- Plot Detection Output Channels ---
    st.subheader("Detection Output Channels")
    x_detection = detection_output.squeeze(0)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6)) # Increased figure size
    titles = ["Confidence", "Range Regression", "Angle Regression"] # More descriptive titles

    for i in range(3):
        channel = x_detection[i].cpu().detach().numpy()
        im2 = axes2[i].imshow(channel, cmap='plasma') # Changed cmap for variety
        axes2[i].set_title(titles[i])
        axes2[i].axis('off')
        fig2.colorbar(im2, ax=axes2[i], orientation='vertical', pad=0.05) # Add colorbar for each subplot

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2) # Close the figure

# --- Streamlit Page Main Function ---
def app_page():
    st.set_page_config(layout="centered", page_title="FFTRadNet Radar Visualization")
    st.title("FFTRadNet Radar Data Visualization")
    st.markdown("Explore the segmentation and detection outputs of the FFTRadNet model for different radar samples.")
    LABELS = pd.read_csv('/home/christophe/ComplexNet/STREAM/labels_CVPR.csv').to_numpy()
    
    CONFIG_PATH = '/home/christophe/ComplexNet/STREAM/config_FFTRadNet_192_56.json'
    CHECKPOINT_PATH = '/home/christophe/ComplexNet/STREAM/FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth'
    ADC_FOLDER = '/home/christophe/RADIalP7/SMALL_DATASET/TEST/ADC/'

    
    net, signal_process_layer, device = load_model(CONFIG_PATH, CHECKPOINT_PATH)

    if net is None or signal_process_layer is None:
        st.error("Model or Signal Processing Layer could not be loaded. Please check paths and error messages above.")
        return

    # Determine min/max idx for slider based on available files
    try:
        adc_files = [f for f in os.listdir(ADC_FOLDER) if f.startswith('raw_adc_') and f.endswith('.npy')]
        if not adc_files:
            st.warning(f"No ADC files found in {ADC_FOLDER}. Please ensure data is present.")
            min_idx, max_idx = 0, 0
        else:
            indices = [int(f.replace('raw_adc_', '').replace('.npy', '')) for f in adc_files]
            min_idx = min(indices)
            max_idx = max(indices)
    except FileNotFoundError:
        st.error(f"ADC data folder not found: {ADC_FOLDER}")
        min_idx, max_idx = 0, 0 # Fallback values
    except Exception as e:
        st.error(f"Error determining ADC file range: {e}")
        min_idx, max_idx = 0, 0

    st.markdown("---")

    # Slider for idx selection
    if min_idx <= max_idx:
        selected_idx = st.slider(
            "Select ADC Data Index:",
            min_value=min_idx,
            max_value=max_idx,
            value=min_idx, # Default to the first available index
            step=1
        )
        st.info(f"Currently viewing data for index: **{selected_idx}**")
        run_inference_and_plot(net, signal_process_layer, device, selected_idx, ADC_FOLDER)
    else:
        st.warning("Could not determine valid range for ADC data indices. Slider not displayed.")
        st.info(f"Please ensure ADC files are in '{ADC_FOLDER}' and named like 'raw_adc_X.npy'.")

# This is the entry point for your Streamlit app
if __name__ == '__main__':
 
    app_page()