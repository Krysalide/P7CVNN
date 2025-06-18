import os
from collections import OrderedDict
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
import streamlit
NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

from Experimental.learnable_fft_wip2 import SignalProcessLayer

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
            raise NameError('Wrong channel angle paraemter !')
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

def main(config, checkpoint_filename,difficult):

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device available: ',device)

    net = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = config['model']['MIMO_output'],
                        channels = config['model']['channels'], 
                        regression_layer = 2, 
                        detection_head = config['model']['DetectionHead'], 
                        segmentation_head = config['model']['SegmentationHead'])

    net.to('cuda')
    print('FFTRadnet initiated')
    checkpoint = torch.load(checkpoint_filename)
    old_state_dict = checkpoint['net_state_dict']

    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        # Remap 'backbone' to 'FPN'
        if k.startswith('backbone.'):
            k = k.replace('backbone.', 'FPN.')

        # Remap 'RAmap_header' to 'RA_decoder'
        if k.startswith('RAmap_header.'):
            k = k.replace('RAmap_header.', 'RA_decoder.')

        # NEW: Remap 'FPN.preproc' to 'FPN.pre_enc'
        if k.startswith('FPN.preproc.'):
            k = k.replace('FPN.preproc.', 'FPN.pre_enc.')

        new_state_dict[k] = v

    try:
        net.load_state_dict(new_state_dict)
        print('Model loaded successfully with remapped keys!')
    except RuntimeError as e:
        print(f"Error after remapping: {e}")

    net.eval()

    signal_process_layer=SignalProcessLayer(use_fft_weights=True).to('cuda')
    adc_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/ADC/'
    idx=2
    sample_adc=np.load(adc_folder+f'raw_adc_{idx}.npy')
        
    batch_sample_adc=torch.tensor(np.expand_dims(sample_adc, axis=0),dtype=torch.complex64)
    signal_processed=signal_process_layer(batch_sample_adc.to('cuda'))

    signal_processed=split_real_imag(signal_processed)
    with torch.set_grad_enabled(False):
            prediction=net(signal_processed)

    detection_output=prediction['Detection']
    
    segmentation_output=prediction['Segmentation']

    spatial_map = segmentation_output.squeeze().cpu().detach().numpy()  # shape: (256, 224)

    # Plot using imshow
    plt.figure(figsize=(6, 6))
    plt.imshow(spatial_map, cmap='gray') 
    plt.colorbar(label='Intensity')
    plt.title("Segmentation FFT output")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

    x = detection_output.squeeze(0)  # Shape: [3, 128, 224]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    for i in range(3):
        channel = x[i].cpu().detach().numpy()  # Shape: [128, 224]
        axes[i].imshow(channel, cmap='viridis')  # or 'gray', 'plasma', etc.
        axes[i].set_title(f"Detection Channel {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='/home/christophe/ComplexNet/STREAM/config_FFTRadNet_192_56.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default='/home/christophe/ComplexNet/STREAM/FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth', type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.difficult)
