import sys
import os
import time
import random
import matplotlib.pyplot as plt


from ComplexUnet import complex_mse_loss
from ComplexUnet import phase_loss
from ComplexUnet import ComplexUNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from p7_utils import list_record_folders,plot_network_loss
from p7_utils import create_dataloaders

def check_gpu_availability():
  """Vérifie si PyTorch peut accéder à un GPU CUDA sur la machine.

  Returns:
    bool: True si un GPU CUDA est disponible et utilisable par PyTorch, False sinon.
  """
  if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"PyTorch has acces to {device_count} GPU(s) CUDA.")
    for i in range(device_count):
      print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda")
    return True,device
  else:
    print("PyTorch n'a pas accès à un GPU CUDA.")
    return False
gpu_ok,device=check_gpu_availability()
if not gpu_ok:
    print('Warning no GPU available')
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


in_channels = 16  
out_channels = 16  
resume_training=False
if resume_training:
   PATH='/home/christophe/ComplexNet/complex_net1.pth'
   model = ComplexUNet(in_channels=16, out_channels=16).to(device)
   model.load_state_dict(torch.load(PATH, weights_only=True))
   save_path='/home/christophe/ComplexNet/complex_net1.pth'
else:
  save_path='/home/christophe/ComplexNet/complex_net_one_run.pth'
  model=ComplexUNet(in_channels=in_channels, out_channels=out_channels).to(device)
# try bigger learning rate

learning_rate = 1e-1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader, val_loader, test_loader = create_dataloaders(
        adc_data,
        rd_data, 
        batch_size=2  
    )

losses=[]
plot_losses=[]
print('------Entering Network Training------------')
total_epochs = 2
print(f"Total epochs: {total_epochs}")
print(f"Batch size: {train_loader.batch_size}")
start_time = time.time()
for epoch in range(total_epochs):
    model.train()
    for batch_data, batch_target in train_loader:
        
        x = batch_data.permute(0, 3, 1, 2).to(device, torch.complex64)
        
        y = batch_target.permute(0, 3, 1, 2).to(device, torch.complex64)
        
        x_re, x_im = x.real, x.imag
        y_re, y_im = y.real, y.imag

        optimizer.zero_grad()
        
        out_complex = model(x)
        out_re, out_im = out_complex.real, out_complex.imag
        #loss = complex_mse_loss(out_re, out_im, y_re, y_im)
        loss=phase_loss(out_complex, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = sum(losses) / len(losses)
    plot_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:.1f}')
    
print(f"Total time for training: {time.time()-start_time:.2f} seconds")
torch.save(model.state_dict(), save_path)
print('------Model Saved------------')
plt.figure(figsize=(10, 6))
plt.plot(plot_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
t_plot=f'Phase Loss, epochs {total_epochs} batch {train_loader.batch_size}'
plt.title(t_plot)
plt.legend()
plt.grid(True)

# Enregistrement du graphique
time_stamp = time.strftime("%m%d-%H%M")
plt.savefig(f'training_loss_{str(time_stamp)}.png')
print("Graphique de la loss enregistré sous 'training_loss.png'")














