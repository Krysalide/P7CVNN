''''
STATUS-RENDU
here we test our pytorch signal process layer.
It is able to produce range doppler maps.
We wanted to test if our layer was able to learn the FFT transform
The answer is positive if one of the fft layer is initiated with exact weights and the other
with random weights.
To achieve this result we had to do some search about loss:
We got good results whit relative losses.
See for example the complex_relative_mse_loss_v3
This work does not have much practical interrest though: exact weights can be set easily.
We produced plots before and after training, some will be shown in the slides.

'''
import sys
import os
import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from ComplexUnet import complex_mse_loss
from ComplexUnet import complex_relative_mse_loss
from ComplexUnet import phase_loss
from ComplexUnet import hybrid_loss


from loss_function_relative import complex_relative_mse_loss_v1
from loss_function_relative import complex_relative_mse_loss_v2
from loss_function_relative import complex_relative_mse_loss_v3
from loss_function_relative import complex_relative_mse_phase_loss


from Experimental.learnable_fft_wip2 import SignalProcessLayerV2

import torch


from torch.optim.lr_scheduler import StepLR
import mlflow


from p7_utils import  check_gpu_availability


from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna
#from data_reader import RadarFFTDataset
from Experimental.data_fft_reader import RadarFFTDataset
from data_reader import split_dataloader

import matplotlib.pyplot as plt
import plotly.graph_objs as go


def plot_hanning_3d_html(tensor, file_name, title="3D Tensor Surface"):
    """
    Creates a 3D surface plot of a PyTorch tensor of shape [512, 256, 1]
    and saves it to an HTML file.
    
    Args:
        tensor (torch.Tensor): Tensor of shape [512, 256, 1].
        file_name (str): File name for the HTML output.
        title (str): Plot title.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # if tensor.shape != (512, 256, 1):
    #     raise ValueError(f"Expected tensor shape [512, 256, 1], but got {tensor.shape}.")

    # Convert to 2D NumPy array
    Z = tensor.squeeze(-1).detach().cpu().numpy()

    # Create coordinate grid
    x = np.arange(Z.shape[1])
    y = np.arange(Z.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create surface plot
    surface = go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='Column',
            yaxis_title='Row',
            zaxis_title='Value',
        )
    )
    fig = go.Figure(data=[surface], layout=layout)

    # Save to HTML
    fig.write_html(f"/home/christophe/ComplexNet/plots/{file_name}")
    print('3D plots saved to: ',f"/home/christophe/ComplexNet/plots/{file_name}")


def plot_tensor_heatmap(tensor,file_name,show_plot, title="Tensor Heatmap", cmap="viridis"):
    """
    Plots a heatmap from a 2D PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): 2D tensor of floats.
    
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
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        assert file_name, 'please specify name of file to be saved'
        plt.savefig('/home/christophe/ComplexNet/plots/'+file_name)
        plt.close()
        return '/home/christophe/ComplexNet/plots/'+file_name

def plot_hanning_window(tensor,file_name,show_plot,title, cmap="viridis"):
    """
    Plots a heatmap for a PyTorch tensor of shape [512, 256, 1].
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
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        save_path='/home/christophe/ComplexNet/plots/'+file_name
        plt.savefig(save_path)
        plt.close()
        return save_path

num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")

gpu_ok,device=check_gpu_availability()


if not gpu_ok:
    sys.exit('No GPU available, will exit')
    
in_channels = 16  # shall remain fixed equal to the number of antennas
out_channels = 16  # same as in_channels
use_fft_weights=False # if true weights will be set with dft matrix weights
resume_training=True

if resume_training:
    print('last trained model will be reloaded')
    model=SignalProcessLayerV2(use_first_fft_weights=True,use_second_fft_weights=False).to(device=device)
    #load_path='/home/christophe/ComplexNet/FFT/signal_process_layer.pth' # v1 signalprocesslayer
    load_path='/home/christophe/ComplexNet/FFT/signal_process_layerV2.pth'
    loaded_state_dict = torch.load(load_path)
    model.load_state_dict(loaded_state_dict)


else:
    print('Will start training from scratch')
    #model=SignalProcessLayer(use_fft_weights=use_fft_weights).to(device=device)
    model=SignalProcessLayerV2(use_first_fft_weights=True,use_second_fft_weights=False).to(device=device)

save_model=True

if save_model:
    print('Model will be saved after training')
else:
    print('Model will not be saved after training')

print('model: ',model.name,' initiated')

model.eval()
range_fft_weights=model.get_range_weights()
plot_tensor_heatmap(tensor=range_fft_weights,title="range_fft_layer_weights_before_train",
                    show_plot=False,file_name='range_fft_weights_before_train.png')
doppler_fft_weights=model.get_doppler_weights()
plot_tensor_heatmap(tensor=doppler_fft_weights,title="dopller_fft_layer_weights_before_train",
                    show_plot=False,file_name='doppler__fft_weights_before_train.png')

plot_hamming=False
if plot_hamming:
    hanning_window_range_coeff=model.get_window_range_coeff()
    plot_hanning_window(tensor=hanning_window_range_coeff,file_name='range_hanning.png'
                    ,title='hanning window range',show_plot=False)
    hanning_window_doppler_coeff=model.get_window_doppler_coeff()
    plot_hanning_window(tensor=hanning_window_doppler_coeff,file_name='hanning_doppler.png',
                    show_plot=False,title='doppler hanning window')
    plot_hanning_3d_html(tensor=hanning_window_doppler_coeff,file_name='HANNING3D.html',title='nightly build')

    plot_hanning_3d_html(tensor=doppler_fft_weights,file_name='3Ddopler_fft_.html',title='fft doppler')
    print()


model.train()
learning_rate = 0.001

batch_size = 2

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,differentiable=False)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


name_optimizer=optimizer.__class__.__name__


print('optimizer used to train: ',name_optimizer)


scheduler=StepLR(step_size=400,gamma=1.1,optimizer=optimizer)

name_scheduler=scheduler.__class__.__name__

print('scheduler: ',name_scheduler)

class LossType(Enum):
    MSE_LOSS = "mse_loss"
    PHASE_LOSS = "phase_loos"
    HYBRID_LOSS = "hybrid_loss"
    RELATIVE_LOSS1="relative_loss1"
    RELATIVE_LOSS2="relative_loss2"
    RELATIVE_LOSS3="relative_loss3"
 
type_loss=LossType.RELATIVE_LOSS2

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
    loss_function=complex_relative_mse_phase_loss
elif type_loss==LossType.RELATIVE_LOSS3:
    loss_function=complex_relative_mse_loss_v3

else:
    raise ValueError("Invalid loss type")

print('Loss type used to train: ',type_loss.value)


print('Entering data loading...')
ratio_test=1/3 # here we take only one sample to train!!! 

    
data_folder='/home/christophe/RADIalP7/SMALL_DATASET/TEST'
assert os.path.exists(data_folder), 'data not found'
element_number=3
assert element_number<61, "number of element is limited to 60 for now"
indices = list(range(element_number)) # specify number of elements

dataset = RadarFFTDataset(data_folder, indices)
print(f"Dataset length: {len(dataset)} (took only {len(indices)} samples)")


# for now all data is splited in train and val, no data for test
train_loader, val_loader, test_loader = split_dataloader(dataset,batch_size=1,train_ratio=ratio_test,val_ratio=ratio_test,test_ratio=ratio_test)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

mlflow.start_run(run_name="random_fft_initialisation_400_epochs_BLR")

# Log parameters
mlflow.log_param("type_of_data",'ADC2FFT')
mlflow.log_param("model_type", model.name)
mlflow.log_param('fft_weights_deterministic',use_fft_weights)
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
epochs = 2000
mlflow.log_param("epochs", epochs)
print(f"Total epochs: {epochs}")
print(f"Batch size: {train_loader.batch_size}")
start_time = time.time()
eval_rate=10
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
        
        
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    mlflow.log_metric("learning_rate_per_epoch", current_lr, step=epoch)

        
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
save_path='/home/christophe/ComplexNet/FFT/signal_process_layerV2.pth'
if save_model:
    torch.save(model.state_dict(), save_path)
    print('------MODEL SAVED------------')
else:
    print('------MODEL NOT SAVED------------')

visualize_train_effects=True
if visualize_train_effects:
    model.eval()
    range_fft_weights=model.get_range_weights()
    post_train_r_fft_path=plot_tensor_heatmap(tensor=range_fft_weights,title="range_fft_layer_weights_after_train",show_plot=False,
                        file_name='range_fft_after_train.png')
    doppler_fft_weights=model.get_doppler_weights()
    post_train_dopller_fft_path=plot_tensor_heatmap(tensor=doppler_fft_weights,title="doppler_fft_layer_weights_after_train",show_plot=False,
                        file_name='doppler_fft_weights_after_train.png')
    hanning_window_range_coeff=model.get_window_range_coeff()
    plot_hanning_window(tensor=hanning_window_range_coeff,file_name='range_hanning_post_train.png'
                    ,title='hanning window range after train',show_plot=False)
    hanning_window_doppler_coeff=model.get_window_doppler_coeff()
    plot_hanning_window(tensor=hanning_window_doppler_coeff,file_name='hanning_doppler_post_train.png',
                    show_plot=False,title='doppler hanning window after train')
    

    

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
mlflow.log_artifact(post_train_r_fft_path)
mlflow.log_artifact(post_train_dopller_fft_path)


mlflow.end_run()


print('------End of Network Training------------')












