# train script
import sys
import os
import time
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt

from ComplexUnet import complex_mse_loss,hybrid_loss
from ComplexUnet import phase_loss
from neuralop.models import FNO2d
from neuralop.losses import H1Loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import mlflow

from p7_utils import list_record_folders,plot_network_loss
from p7_utils import create_dataloaders, check_gpu_availability
from p7_utils import normalize_complex_amplitude

from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna
from data_reader import RadarDataset,RadarDatasetV2
from data_reader import load_data
from data_reader import split_dataloader


num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")

gpu_ok,_=check_gpu_availability()
device='cuda'
if not gpu_ok:
    
    sys.exit('No GPU available, exiting')
    
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

model_type='fno'

if model_type=='fno':
    
    
    if resume_training:
        model=FNO2d(complex_data=True,in_channels=in_channels,out_channels=out_channels,n_modes_height=256,n_modes_width=512,hidden_channels=16).to(device)
        PATH='/home/christophe/ComplexNet/FNO/fno.pth'
        model.load_state_dict(torch.load(PATH, weights_only=False))
        print('Model loaded from', PATH)
        save_path='/home/christophe/ComplexNet/FNO/fno.pth'
    else:
        save_path='/home/christophe/ComplexNet/FNO/fno_one_run.pth' 
        model=FNO2d(complex_data=True,in_channels=16,out_channels=16,n_modes_height=256,n_modes_width=512,hidden_channels=16).to(device)

else:
    raise ValueError('not valid model')


learning_rate = 0.1
print('learning rate: ',learning_rate)
step_size = 10
gamma = 0.95


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
    print('Warning loss function seems to be non convex!!!')

elif type_loss==LossType.HYBRID_LOSS:
    loss_function = hybrid_loss
else:
    raise ValueError("Invalid loss type")

full_data=True
if not full_data:
    # use if you want a small set of data
    sequence = 'RECORD@2020-11-21_11.54.31'
    data_folder = f'/media/christophe/backup/DATARADIAL/{sequence}'
    indices = list(range(50)) # specify number of elements

    dataset = RadarDataset(data_folder, indices)
    print(f"Dataset length: {len(dataset)} (took only {len(indices)} samples from sequence: {sequence})")

else:
    data_folder='/media/christophe/backup/DATARADIAL'
    dataset=RadarDatasetV2(data_folder,recursive=True)
    print(f"Dataset length: {len(dataset)}, gathered all data available from folder: {data_folder} ")
    


train_loader, val_loader, test_loader = split_dataloader(dataset,batch_size=8)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

mlflow.start_run()

# Log parameters
mlflow.log_param("model_type", 'fno_model')
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("optimizer", name_optimizer)
mlflow.log_param("scheduler", name_scheduler)
mlflow.log_param("step_size", step_size)
mlflow.log_param("gamma", gamma)
mlflow.log_param("batch_size", train_loader.batch_size)

mlflow.log_param("loss_function", type_loss.value)
mlflow.log_param("resume_training", resume_training)
mlflow.log_param("number of training samples", len(train_loader.dataset))
mlflow.log_param("number of validation samples", len(val_loader.dataset))
mlflow.log_param("number of test samples", len(test_loader.dataset))

losses=[]
plot_losses=[]
val_mse_history = []
val_phase_history = []
print('------Entering Network Training------------')
epochs =30
mlflow.log_param("epochs", epochs)
print(f"Total epochs: {epochs}")
print(f"Batch size: {train_loader.batch_size}")
start_time = time.time()
eval_rate=5
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
    if (epoch+1) % eval_rate == 0:
        model.eval()
        mses=[]
        phases=[]
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                x = batch_data.permute(0, 3, 1, 2).to(device, torch.complex64)
                y = batch_target.permute(0, 3, 1, 2).to(device, torch.complex64)
                out_complex = model(x)
                
                mse_per_antenna=complex_mse_per_antenna(out_complex, y).mean().item()
                mses.append(mse_per_antenna)
                
                
                phase_error=phase_error_per_antenna(out_complex, y).mean().item()
                phases.append(phase_error)
                
        avg_mse = sum(mses) / len(mses)
        avg_phase = sum(phases) / len(phases)
        val_mse_history.append(avg_mse)
        val_phase_history.append(avg_phase)
        mlflow.log_metric("val_mse_per_antenna", avg_mse, step=epoch) # check if step is ok?
        mlflow.log_metric("val_phase_error_per_antenna", avg_phase, step=epoch)
        print('------Validation Results------------')
        print(f"Epoch [{epoch+1}/{epochs}], MSE per antenna: {avg_mse:.4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Phase error per antenna: {avg_phase:.4f}")
        print('------------------------------------')


print(f"Total time per epoch: {(time.time()-start_time)/epochs:.2f} seconds")
if save_model:
    torch.save(model.state_dict(), save_path)
    print('------MODEL SAVED------------')
else:
    print('------MODEL NOT SAVED------------')


plt.figure(figsize=(10, 6))
plt.plot(plot_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plot_title=f'Model: {model_type} epochs {epochs} batch {train_loader.batch_size} samples {len(train_loader.dataset)}'
plt.title(plot_title)
plt.legend()
plt.grid(True)


time_stamp = time.strftime("%m%d-%H%M")
plot_path = f'training_loss_fno_{str(time_stamp)}.png'
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
plot_path2 = f'validation_metrics_fno_{str(time_stamp)}.png'
plt.savefig(plot_path2)
plt.close()


mlflow.log_artifact(plot_path)
mlflow.log_artifact(plot_path2)
mlflow.end_run()
# plots are already in mlflow
os.remove(plot_path)
os.remove(plot_path2)

print('------End of Network Training------------')

# 100 samples 36 seconds
# 250 smaples 86 seconds














