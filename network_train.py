import sys
import os
import time
import random
import matplotlib.pyplot as plt

from ComplexUnet import complex_mse_loss
from ComplexUnet import phase_loss
from ComplexUnet import ComplexUNet
from ComplexCardoidUnet import ComplexUNetCardioid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import mlflow
import numpy as np
from p7_utils import list_record_folders,plot_network_loss
from p7_utils import create_dataloaders, check_gpu_availability
from p7_utils import normalize_complex_amplitude
from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna

gpu_ok,device=check_gpu_availability()
if not gpu_ok:
    #print('Warning no GPU available')
    sys.exit('No GPU available, exiting')
    
disk_adress='/media/christophe/backup/DATASET/'

folders_adc=list_record_folders('/media/christophe/backup/DATASET/ADC/')
#print(folders_adc)
folders_range_doppler=list_record_folders('/media/christophe/backup/DATASET/RDGD/')
#print(folders_range_doppler)

# TODO retrieve all data from folders
adc_folder1=folders_adc[0]
rd_folder1=folders_range_doppler[0]
print(adc_folder1)
print(rd_folder1)
adc_dat_file=adc_folder1+'/complex_data.npy'
rd_dat_file=rd_folder1+'/rd_maps.npy'
start_time = time.time()
adc_data = np.load(adc_dat_file)
print(adc_data.shape)
print((time.time()-start_time)," seconds to load adc data")
start_time = time.time()
rd_data = np.load(rd_dat_file)
print(rd_data.shape)
print(time.time()-start_time," seconds to load rd data")

in_channels = 16  # shall remain fixed equal to the number of antennas
out_channels = 16  # same as in_channels
resume_training=True
cardioid_model=True

if cardioid_model:
    model = ComplexUNetCardioid(in_channels=in_channels, out_channels=out_channels).to(device)
    print(model.name)
    if resume_training:
        PATH='/home/christophe/ComplexNet/complex_cardioid_net.pth'
        model.load_state_dict(torch.load(PATH, weights_only=True))
        print('Model loaded from', PATH)
        save_path='/home/christophe/ComplexNet/complex_cardioid_net.pth'
    else:
        save_path='/home/christophe/ComplexNet/complex_cardioid_net_one_run.pth'
        model=ComplexUNetCardioid(in_channels=in_channels, out_channels=out_channels).to(device)
else:
   if resume_training:
    PATH='/home/christophe/ComplexNet/complex_net1.pth'
    model = ComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    print(model.name)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    save_path='/home/christophe/ComplexNet/complex_net1.pth'
   else:
    save_path='/home/christophe/ComplexNet/complex_net_one_run.pth'
    model=ComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)


learning_rate = 1e-1
step_size = 5
gamma = 0.96
batch_size = 2
epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

name_optimizer=optimizer.__class__.__name__

scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
name_scheduler=scheduler.__class__.__name__


train_loader, val_loader, test_loader = create_dataloaders(
        adc_data,
        rd_data, 
        batch_size=batch_size
    )
mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("optimizer", name_optimizer)
mlflow.log_param("scheduler", name_scheduler)
mlflow.log_param("step_size", step_size)
mlflow.log_param("gamma", gamma)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)

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
        
        x_re, x_im = x.real, x.imag
        y_re, y_im = y.real, y.imag

        optimizer.zero_grad()
        
        out_complex = model(x)
        out_re, out_im = out_complex.real, out_complex.imag
        loss = complex_mse_loss(out_re, out_im, y_re, y_im)
        #loss=phase_loss(out_complex, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
        
    avg_loss = sum(losses) / len(losses)
    plot_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.1f}')



print(f"Total time for training: {time.time()-start_time:.2f} seconds")
torch.save(model.state_dict(), save_path)
print('------Model Saved------------')
mlflow.end_run()
plt.figure(figsize=(10, 6))
plt.plot(plot_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
t_plot=f'Phase Loss, epochs {epochs} batch {train_loader.batch_size}'
plt.title(t_plot)
plt.legend()
plt.grid(True)

# Enregistrement du graphique
time_stamp = time.strftime("%m%d-%H%M")
plt.savefig(f'training_loss_{str(time_stamp)}.png')
print("Graphique de la loss enregistré sous:")
print(f'training_loss_{str(time_stamp)}.png')
print('------End of Network Training------------')














