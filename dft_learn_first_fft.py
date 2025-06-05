# train script

# TODO
# continue testing loss functions, see other tactics
# good learning rate seem to be vary small -> maybe other optimizer?
# increase size of dataset done
# have a look to validation metrics
# log with mlflow the loss function (enum?)
# add possibility to resume training 


import matplotlib.pyplot as plt

def plot_tensor_heatmap(tensor,file_name,show_plot, title="Tensor Heatmap", cmap="viridis"):
    """
    Plots a heatmap from a 2D PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): 2D tensor of floats.
        title (str): Title of the plot.
        cmap (str): Colormap for the heatmap (e.g., 'viridis', 'hot', 'coolwarm').
        show_values (bool): If True, overlays float values in cells.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D")

    tensor_np = tensor.detach().cpu().numpy()  # convert to NumPy for matplotlib

    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(tensor_np, cmap=cmap, aspect='auto')
    plt.colorbar(heatmap)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")

    

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        assert file_name
        plt.savefig('/home/christophe/ComplexNet/plots/'+file_name)
        plt.close()

def plot_hanning_window(tensor, title="Heatmap", cmap="viridis"):
    """
    Plots a heatmap for a PyTorch tensor of shape [512, 256, 1].

    Args:
        tensor (torch.Tensor): Tensor of shape [512, 256, 1].
        title (str): Plot title.
        cmap (str): Colormap used for the heatmap.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    if tensor.shape != (512, 256, 1):
        raise ValueError(f"Expected tensor shape [512, 256, 1], but got {tensor.shape}.")

    tensor_2d = tensor.squeeze(-1).detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(tensor_2d, cmap=cmap, aspect='auto')
    plt.colorbar(heatmap)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()

import sys
import os
import time
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt

from ComplexUnet import complex_mse_loss
from ComplexUnet import complex_relative_mse_loss
from ComplexUnet import phase_loss
from ComplexUnet import hybrid_loss

# wip to be tested
from loss_function_relative import complex_relative_mse_loss_v1,complex_relative_mse_loss_v2
from loss_function_relative import complex_relative_mse_loss_v3

from Experimental.learnable_fft_wip2 import SignalProcessLayer

from ComplexUnet import visualize_complex_norm,visualize_complex_plane,visualize_complex_weights

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import mlflow

from p7_utils import list_record_folders,plot_network_loss
from p7_utils import create_dataloaders, check_gpu_availability
from p7_utils import normalize_complex_amplitude

from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna
#from data_reader import RadarFFTDataset
from Experimental.data_fft_reader import RadarFFTDataset
from data_reader import split_dataloader


num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")

gpu_ok,_=check_gpu_availability()
device='cuda'
if not gpu_ok:
    sys.exit('No GPU available, exiting')
    
in_channels = 16  # shall remain fixed equal to the number of antennas
out_channels = 16  # same as in_channels
resume_training=False

if resume_training:
    print('Resume training')
    raise NotImplementedError
else:
    print('Will start training from scratch')
    model=SignalProcessLayer().to(device=device)

save_model=True
if save_model:
    print('Model will be saved after training')
else:
    print('Model will not be saved after training')

print('model: ',model.name,' initiated')

# TODO check if code ok:
for param in model.hamming1.parameters():
    param.requires_grad = False
for param in model.hamming2.parameters():
    param.requires_grad = False


for param in model.first_fft_layer.parameters():
    param.requires_grad = True
for param in model.second_fft_layer.parameters():
    param.requires_grad = True

for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")


model.eval()
range_fft_weights=model.get_range_weights()
plot_tensor_heatmap(tensor=range_fft_weights,title="range_fft_layer_weights_before_train",
                    show_plot=False,file_name='fft_weights_before_train.png')
doppler_fft_weights=model.get_doppler_weights()
plot_tensor_heatmap(tensor=doppler_fft_weights,title="range_fft_layer_weights_before_train",
                    show_plot=False,file_name='fft_doppler_weights_before_train.png')

hanning_window_range_coeff=model.get_window_range_coeff()
plot_hanning_window(hanning_window_range_coeff)
hanning_window_doppler_coeff=model.get_window_doppler_coeff()
plot_hanning_window(hanning_window_doppler_coeff)
print()

model.train()
learning_rate = 1e-5

batch_size = 2


#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,differentiable=False)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


name_optimizer=optimizer.__class__.__name__


print('optimizer for that run: ',name_optimizer)

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                #  factor=0.95, patience=25, 
                                                #  min_lr=1e-7,
                                                # threshold=100,threshold_mode='abs')


scheduler=StepLR(step_size=10,gamma=0.85,optimizer=optimizer)


name_scheduler=scheduler.__class__.__name__

print('scheduler: ',name_scheduler)

class LossType(Enum):
    MSE_LOSS = "mse_loss"
    PHASE_LOSS = "phase_loos"
    HYBRID_LOSS = "hybrid_loss"
    RELATIVE_LOSS1="relative_loss1"
    RELATIVE_LOSS2="relative_loss2"
    RELATIVE_LOSS3="relative_loss3"
 
type_loss=LossType.RELATIVE_LOSS1

if type_loss==LossType.MSE_LOSS:
    loss_function = complex_mse_loss
elif type_loss==LossType.PHASE_LOSS:
    loss_function = phase_loss
    print('Warning loss function seems to be non convex!!!')
elif type_loss==LossType.HYBRID_LOSS:
    loss_function = hybrid_loss
elif type_loss==LossType.RELATIVE_LOSS1:
    loss_function=complex_relative_mse_loss_v1
elif type_loss==LossType.RELATIVE_LOSS2:
    loss_function=complex_relative_mse_loss_v2
elif type_loss==LossType.RELATIVE_LOSS3:
    loss_function=complex_relative_mse_loss_v3

else:
    raise ValueError("Invalid loss type")

print('Loss type: ',type_loss.value)


print('data loading...')
full_data=False
if not full_data:
    
    data_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST'
    assert os.path.exists(data_folder), 'data not found'
    element_number=60
    assert element_number<61, f"number of element is limited to 60"
    indices = list(range(element_number)) # specify number of elements

    dataset = RadarFFTDataset(data_folder, indices)
    print(f"Dataset length: {len(dataset)} (took only {len(indices)} samples)")

else:
    raise NotImplementedError('recursive dataset not yet built for one fft')
    data_folder='/media/christophe/backup/DATARADIAL'
    dataset=RadarDatasetV2(data_folder,recursive=True)
    print(f"Dataset length: {len(dataset)}, gathered all data available from folder: {data_folder} ")


# for now all data is splited in train and val, no data for test
train_loader, val_loader, test_loader = split_dataloader(dataset,batch_size=4,train_ratio=0.8,val_ratio=0.19,test_ratio=0.01)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

mlflow.start_run()

# Log parameters
mlflow.log_param("type_of_data",'ONEFFT')
mlflow.log_param("model_type", 'FFT_DFT_LAYER')
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("optimizer", name_optimizer)
mlflow.log_param("scheduler", name_scheduler)
mlflow.log_param('loss function',type_loss.value)

mlflow.log_param("batch_size", batch_size)

mlflow.log_param("resume_training", resume_training)
mlflow.log_param("number of training samples", len(train_loader.dataset))
mlflow.log_param("number of validation samples", len(val_loader.dataset))
mlflow.log_param("number of test samples", len(test_loader.dataset))

losses=[]
plot_losses=[]
val_mse_history = []
val_phase_history = []
val_loss_history=[]
print('------Entering Network Training------------')
epochs = 50
mlflow.log_param("epochs", epochs)
print(f"Total epochs: {epochs}")
print(f"Batch size: {train_loader.batch_size}")
start_time = time.time()
eval_rate=25
for epoch in range(epochs):
    model.train()
    for batch_data, batch_target in train_loader:
        
        
        x = batch_data.to(device, torch.complex64)
        y = batch_target.to(device, torch.complex64)
    
        optimizer.zero_grad()
        out_complex = model(x)
    
        loss=loss_function(out_complex, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    
        
    avg_loss = sum(losses) / len(losses)
    plot_losses.append(avg_loss)
    mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    if (epoch+1) % eval_rate == 0:
        model.eval()
        mses=[]
        phases=[]
        val_losses=[]
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                x = batch_data.to(device, torch.complex64)
                y = batch_target.to(device, torch.complex64)
                out_complex = model(x)
                loss_val=loss_function(out_complex,y).item() # same as train loss
                val_losses.append(loss_val)

                mse_per_antenna=complex_mse_per_antenna(out_complex, y).mean().item()
                mses.append(mse_per_antenna)
                
                phase_error=phase_error_per_antenna(out_complex, y).mean().item()
                phases.append(phase_error)
                
        avg_mse = sum(mses) / len(mses)
        avg_phase = sum(phases) / len(phases)
        avg_loss=sum(val_losses)/len(val_losses)
        
        val_mse_history.append(avg_mse)
        val_phase_history.append(avg_phase)
        val_loss_history.append(avg_loss)
        mlflow.log_metric("val_mse_per_antenna", avg_mse, step=epoch) # check if step is ok?
        mlflow.log_metric("val_phase_error_per_antenna", avg_phase, step=epoch)
        mlflow.log_metric("Val_loss",avg_loss,step=epoch)
        print('------Validation Results------------')
        
        print(f"Epoch [{epoch+1}/{epochs}], Val loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], MSE per antenna: {avg_mse:.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Phase error per antenna: {avg_phase:.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Phase error per antenna: {avg_loss:.1f}")
        print('------------------------------------')


print(f"Total time per epoch: {(time.time()-start_time)/epochs:.2f} seconds")
save_path='/home/christophe/ComplexNet/FFT/signal_process_layer.pth'
if save_model:
    torch.save(model.state_dict(), save_path)
    print('------MODEL SAVED------------')
else:
    print('------MODEL NOT SAVED------------')

visualize=True
if visualize:
    model.eval()
    range_fft_weights=model.get_range_weights()
    plot_tensor_heatmap(tensor=range_fft_weights,title="range_fft_layer_weights_after_train",show_plot=False,
                        file_name='range_fft_after_train.png')
    doppler_fft_weights=model.get_doppler_weights()
    plot_tensor_heatmap(tensor=doppler_fft_weights,title="doppler_fft_layer_weights_after_train",show_plot=False,
                        file_name='doppler_fft_weights_after_train.png')
    

plt.figure(figsize=(10, 6))
plt.plot(plot_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plot_title=f'Model: fft_layer epochs {epochs} batch {train_loader.batch_size} samples {len(train_loader.dataset)}'
plt.title(plot_title)
plt.legend()
plt.grid(True)


time_stamp = time.strftime("%m%d-%H%M")
plot_path = f'training_loss_{str(time_stamp)}.png'
plt.savefig(plot_path)
plt.close()

eval_epochs = list(range(eval_rate - 1, epochs, eval_rate))

plt.figure(figsize=(12, 5))

# MSE plot
plt.subplot(1, 2, 1)
plt.plot(eval_epochs, val_mse_history, marker='o', label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE per antenna')
plt.title('Validation MSE Over Epochs')
plt.grid(True)
plt.legend()

# Phase Error plot
plt.subplot(1, 2, 2)
plt.plot(eval_epochs, val_phase_history, marker='x', color='orange', label='Validation Phase Error')
plt.xlabel('Epoch')
plt.ylabel('Phase Error per antenna')
plt.title('Validation Phase Error Over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plot_path2 = f'validation_metrics_{str(time_stamp)}.png'
plt.savefig(plot_path2)
plt.close()


mlflow.log_artifact(plot_path)
mlflow.log_artifact(plot_path2)

# if model_type==NetType.ONE_LAYER:

#     mlflow.log_artifact(path_complex_weight)
#     mlflow.log_artifact(path_complex_norm)

mlflow.end_run()
#os.remove(plot_path)
#os.remove(plot_path2)
# if model_type==NetType.ONE_LAYER:
#     os.remove(path_complex_weight)
#     os.remove(path_complex_norm)

print('------End of Network Training------------')

# 100 samples 36 seconds
# 250 smaples 86 seconds














