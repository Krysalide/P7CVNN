from enum import Enum
import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from ComplexUnet import complex_mse_loss,hybrid_loss
from ComplexUnet import phase_loss
from ComplexUnet import ComplexUNet
from ComplexUnet import SmallComplexUNet
from ComplexCardoidUnet import ComplexUNetCardioid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import mlflow

from p7_utils import list_record_folders,plot_network_loss
from p7_utils import create_dataloaders, check_gpu_availability
from p7_utils import normalize_complex_amplitude
from p7_utils import get_physical_cpu_count
from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna
from data_reader import RadarDataset
from data_reader import load_data
from data_reader import split_dataloader
get_physical_cpu_count()
num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")
gpu_ok,device=check_gpu_availability()
if not gpu_ok:
    
    sys.exit('No GPU available, exiting')
    


sequence = 'RECORD@2020-11-21_11.54.31'
save_folder = f'/media/christophe/backup/DATARADIAL/{sequence}'


in_channels = 16  # shall remain fixed equal to the number of antennas
out_channels = 16  # same as in_channels
resume_training=True
if resume_training:
    print('Resume training')
else:
    print('Will start training from scratch')
save_model=True
if save_model:
    print('Model will be saved after training')
else:
    print('Model will not be saved after training')

class NetType(Enum):
    UNET = "complex_unet"
    CARDIOID_UNET = "complex_cardioid_unet"
    SMALL_UNET = "complex_small_unet"
    
#cardioid_model=True
model_type=NetType.SMALL_UNET

if model_type==NetType.CARDIOID_UNET:
    model = ComplexUNetCardioid(in_channels=in_channels, out_channels=out_channels).to(device)
    
    if resume_training:
        PATH='/home/christophe/ComplexNet/complex_cardioid_net.pth'
        model.load_state_dict(torch.load(PATH, weights_only=True))
        print('Model loaded from', PATH)
        save_path='/home/christophe/ComplexNet/complex_cardioid_net.pth'
    else:
        save_path='/home/christophe/ComplexNet/complex_cardioid_net_one_run.pth'
        model=ComplexUNetCardioid(in_channels=in_channels, out_channels=out_channels).to(device)
elif model_type==NetType.UNET:
    if resume_training:
        PATH='/home/christophe/ComplexNet/complex_net1.pth'
        model = ComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)
        print(model.name)
        model.load_state_dict(torch.load(PATH, weights_only=True))
        save_path='/home/christophe/ComplexNet/complex_net1.pth'
    else:
        save_path='/home/christophe/ComplexNet/complex_net_one_run.pth'
        model=ComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)
elif model_type==NetType.SMALL_UNET:
    if resume_training:
        PATH='/home/christophe/ComplexNet/small_complex_net.pth'
        model = SmallComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)
        
        model.load_state_dict(torch.load(PATH, weights_only=True))
        save_path='/home/christophe/ComplexNet/small_complex_net.pth'
    else:
        save_path='/home/christophe/ComplexNet/small_complex_net_one_run.pth'
        model=SmallComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    
else:
    raise ValueError("Invalid model type")

print(model.name)
learning_rate = 0.1
step_size = 10
gamma = 0.95
batch_size = 2
epochs = 50

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

name_optimizer=optimizer.__class__.__name__

scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
name_scheduler=scheduler.__class__.__name__

class LossType(Enum):
    MSE_LOSS = "mse_loss"
    PHASE_LOSS = "phase_loos"
    HYBRID_LOSS = "hybrid_loss"
    
type_loss=LossType.MSE_LOSS

if type_loss==LossType.MSE_LOSS:
    loss_function = complex_mse_loss
elif type_loss==LossType.PHASE_LOSS:
    loss_function = phase_loss
elif type_loss==LossType.HYBRID_LOSS:
    loss_function = hybrid_loss
else:
    raise ValueError("Invalid loss type")

# train_loader, val_loader, test_loader = create_dataloaders(
#         adc_data,
#         rd_data, 
#         batch_size=batch_size
#     )
# train_loader = load_data(
#     save_folder=save_folder,
#     indices=list(range(250))
#
indices = list(range(250))

dataset = RadarDataset(save_folder, indices)
print(f"Dataset length: {len(dataset)}")
#dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,pin_memory=True)
train_loader, val_loader, test_loader = split_dataloader(dataset)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
mlflow.start_run()

# Log parameters
mlflow.log_param("model_type", model.name)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("optimizer", name_optimizer)
mlflow.log_param("scheduler", name_scheduler)
mlflow.log_param("step_size", step_size)
mlflow.log_param("gamma", gamma)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)
mlflow.log_param("loss_function", type_loss.value)
mlflow.log_param("resume_training", resume_training)

losses=[]
plot_losses=[]
print('------Entering Network Training------------')

print(f"Total epochs: {epochs}")
print(f"Batch size: {train_loader.batch_size}")
start_time = time.time()
eval_rate=10
for epoch in range(epochs):
    model.train()
    for batch_data, batch_target in train_loader:
        
        
        x = batch_data.permute(0, 3, 1, 2).to(device, torch.complex64)
        y = batch_target.permute(0, 3, 1, 2).to(device, torch.complex64)
    
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

print(f"Total time per epoch: {(time.time()-start_time)/epochs:.2f} seconds")
if save_model:
    torch.save(model.state_dict(), save_path)
    print('------MODEL SAVED------------')
else:
    print('------MODEL NOT SAVED------------')
#torch.save(model.state_dict(), save_path)
print('------Model Saved------------')
mlflow.end_run()
plt.figure(figsize=(10, 6))
plt.plot(plot_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
t_plot=f'Model: {model_type} epochs {epochs} batch {train_loader.batch_size} samples {len(train_loader.dataset)}'
plt.title(t_plot)
plt.legend()
plt.grid(True)

# Enregistrement du graphique
time_stamp = time.strftime("%m%d-%H%M")
plt.savefig(f'training_loss_{str(time_stamp)}.png')
print("Graphique de la loss enregistré sous:")
print(f'training_loss_{str(time_stamp)}.png')
print('------End of Network Training------------')

# 100 samples 36 seconds
# 250 smaples 86 seconds














