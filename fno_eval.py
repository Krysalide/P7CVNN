import time
import numpy as np
import sys
import torch
import torch.nn as nn

from data_reader import RadarDataset,split_dataloader
from neuralop.models import FNO2d 


from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna


import plotly.graph_objects as go
import matplotlib.pyplot as plt
interactive = False
graph_2D=False
graph_3D=False

# conda env: complex_net
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

model_evaluated='fno'
in_channels=16
out_channels=16
if model_evaluated=='fno':
        model=FNO2d(complex_data=True,in_channels=in_channels,out_channels=out_channels,n_modes_height=256,n_modes_width=512,hidden_channels=16).to(device)
        PATH='/home/christophe/ComplexNet/FNO/fno.pth'
        model.load_state_dict(torch.load(PATH, weights_only=False))
        print('Model loaded from', PATH)
    
else:
    raise ValueError("Invalid model name. Choose 'ComplexUNet' or 'ComplexUNetCardioid'.")


model.eval()
print('Model loaded from', PATH)

#print_model_layers(model)

sequence = 'RECORD@2020-11-21_11.54.31'
save_folder = f'/media/christophe/backup/DATARADIAL/{sequence}'
indices = list(range(300))

dataset = RadarDataset(save_folder, indices)
print(f"Dataset length: {len(dataset)}")

train_loader, val_loader, test_loader = split_dataloader(dataset)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")


print("=== Dataloaders created ===")
print("Evaluating model on test set...")
test_loss = 0.0
test_mse = 0.0
test_mae = 0.0
test_phase = 0.0
test_rel = 0.0
test_real_mse = 0.0
test_imag_mse = 0.0
test_count = 0 
got_sample=False 
with torch.no_grad():
    for batch_data, batch_target in test_loader:
        x = batch_data.permute(0, 3, 1, 2).to(device, torch.complex64)
        y = batch_target.permute(0, 3, 1, 2).to(device, torch.complex64)
        if not got_sample:
            sample_adc_frame = x[0]
            sample_rd_map = y[0]
            print("Sample ADC frame shape:", sample_adc_frame.shape)
            print("Sample RD map shape:", sample_rd_map.shape)
            got_sample=True
        x_re, x_im = x.real, x.imag
        y_re, y_im = y.real, y.imag

        out_complex = model(x)
        out_re, out_im = out_complex.real, out_complex.imag

        mse = complex_mse_per_antenna(out_complex, y)
        mae = complex_mae_per_antenna(out_complex, y)
        phase = phase_error_per_antenna(out_complex, y)
        rel = relative_error_per_antenna(out_complex, y)
        real_mse, imag_mse = real_imag_mse_per_antenna(out_complex, y)

        test_mse += mse.sum().item()
        test_mae += mae.sum().item()
        test_phase += phase.sum().item()
        test_rel += rel.sum().item()
        test_real_mse += real_mse.sum().item()
        test_imag_mse += imag_mse.sum().item()

        test_count += x.size(0)  # Compte le nombre de frames traitées

print('Number of frames processed:', test_count)
print("=== Résultats de l'évaluation ===")
print(f"MSE moyen par antenne: {test_mse / test_count:.4f}")
print(f"MAE moyen par antenne: {test_mae / test_count:.4f}")
print(f"Erreur de phase moyenne par antenne: {test_phase / test_count:.4f}")
print(f"Erreur relative moyenne par antenne: {test_rel / test_count:.4f}")
print(f"MSE réelle moyenne par antenne: {test_real_mse / test_count:.4f}")
print(f"MSE imaginaire moyenne par antenne: {test_imag_mse / test_count:.4f}")
print("=== Fin de l'évaluation du modèle ===")


rd_map_predicted = model(sample_adc_frame)
print(rd_map_predicted.shape)
rd_map_predicted=torch.squeeze(rd_map_predicted)
print(rd_map_predicted.shape)



# we run over all the antennas
# and plot the results
for i in range(16):
    if not graph_2D:
        continue
    print(f"\n--- Antenne {i} ---")

    
    sample_rd_map_antenna = sample_rd_map[i] # ground truth
    antenna_rd_predicted = rd_map_predicted[i] # predicted


    # Phase
    phase_rd_gt_torch = torch.angle(sample_rd_map_antenna)
    phase_rd_torch = torch.angle(antenna_rd_predicted)
    # magnitude
    mag_rd_map_gt_torch = torch.abs(sample_rd_map_antenna)
    mag_rd_torch = torch.abs(antenna_rd_predicted)

    # Numpy conversion
    phase_adc_np = phase_rd_gt_torch.cpu().numpy()
    phase_rd_np = phase_rd_torch.cpu().detach().numpy()
    mag_adc_np = mag_rd_map_gt_torch.cpu().numpy()
    mag_rd_np = mag_rd_torch.cpu().detach().numpy()

    # ---------- FIGURE 1 : Phases ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(phase_adc_np, cmap='viridis')
    axes[0].set_title(f'GT Phase antenna_{i}_adc')
    axes[0].set_xlabel('Doppler')
    axes[0].set_ylabel('Range')
    fig.colorbar(im1, ax=axes[0], label='Phase (radians)')

    im2 = axes[1].imshow(phase_rd_np, cmap='viridis')
    axes[1].set_title(f'Pred Phase antenna_{i}_rd')
    axes[1].set_xlabel('Doppler')
    axes[1].set_ylabel('Range')
    fig.colorbar(im2, ax=axes[1], label='Phase (radians)')

    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(f'./FNO/plot2D/phase_adc_rd_antenna_{i}.png', dpi=300)
        plt.close(fig)
    # ---------- FIGURE 2 : Différence ----------
    phase_difference_np = phase_adc_np - phase_rd_np

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    im_diff = ax2.imshow(phase_difference_np, cmap='coolwarm')
    ax2.set_title(f'Différence de Phase (antenna_{i}_adc - antenna_{i}_rd)')
    ax2.set_xlabel('Doppler')
    ax2.set_ylabel('Range')
    fig2.colorbar(im_diff, ax=ax2, label='Différence de Phase (radians)')

    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(f'./FNO/plot2D/phase_difference_antenna_{i}.png', dpi=300)
        plt.close(fig2)

    # ---------- FIGURE 3 : Magnitudes ----------
    fig3, axes = plt.subplots(1, 2, figsize=(12, 6))
    im3 = axes[0].imshow(mag_adc_np, cmap='viridis')
    axes[0].set_title(f'Magnitude antenna_{i}_adc')
    axes[0].set_xlabel('Doppler')
    axes[0].set_ylabel('Range')
    fig3.colorbar(im3, ax=axes[0], label='Magnitude')
    im4 = axes[1].imshow(mag_rd_np, cmap='viridis')
    axes[1].set_title(f'Predicted Magnitude antenna_{i}_rd')
    axes[1].set_xlabel('Doppler')
    axes[1].set_ylabel('Range')
    fig3.colorbar(im4, ax=axes[1], label='Magnitude')
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(f'./FNO/plot2D/magnitude_adc_rd_antenna_{i}.png', dpi=300)
        plt.close(fig3)
# ---------- FIGURE 4 : Différence Magnitudes ----------
    mag_difference_np = mag_adc_np - mag_rd_np
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    im_diff_mag = ax4.imshow(mag_difference_np, cmap='coolwarm')
    ax4.set_title(f'Différence de Magnitude (antenna_{i}_adc - antenna_{i}_rd)')
    ax4.set_xlabel('Doppler')
    ax4.set_ylabel('Range')
    fig4.colorbar(im_diff_mag, ax=ax4, label='Différence de Magnitude')
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(f'./FNO/plot2D/magnitude_difference_antenna_{i}.png', dpi=300)
        plt.close(fig4)


for i in range(16):
    if not graph_3D:
        continue
    print(f"\n--- Antenne {i} ---")


    rd_map_gt = sample_rd_map[i]
    antenna_rd_predicted = rd_map_predicted[i]

    print("ADC shape:", rd_map_gt.shape, "type:", type(rd_map_gt))
    print("RD  shape:", antenna_rd_predicted.shape, "type:", type(antenna_rd_predicted))

    # Phase
    phase_adc_np = torch.angle(rd_map_gt).cpu().numpy()
    phase_rd_np = torch.angle(antenna_rd_predicted).cpu().detach().numpy()
    phase_diff_np = phase_adc_np - phase_rd_np
    # magnitude
    mag_adc_np = torch.abs(rd_map_gt).cpu().numpy()
    mag_rd_np = torch.abs(antenna_rd_predicted).cpu().detach().numpy()
    mag_diff_np = mag_adc_np - mag_rd_np

    
    fig_adc = go.Figure(data=[go.Surface(z=phase_adc_np)])
    fig_adc.update_layout(
        title=f'Phase ADC - Antenna {i}',
        scene=dict(
            xaxis_title='Doppler',
            yaxis_title='Range',
            zaxis_title='Phase (rad)'
        )
    )
    fig_adc.write_html(f'./FNO/plot3D/phase_adc_antenna_{i}.html')
    #fig_adc.show()

    # ---------- FIGURE 2 : Phase RD ----------
    fig_rd = go.Figure(data=[go.Surface(z=phase_rd_np)])
    fig_rd.update_layout(
        title=f'Phase RD (GT) - Antenna {i}',
        scene=dict(
            xaxis_title='Doppler',
            yaxis_title='Range',
            zaxis_title='Phase (rad)'
        )
    )
    #fig_rd.show()
    fig_rd.write_html(f'./FNO/plot3D/phase_rd_antenna_{i}.html')

    # ---------- FIGURE 3 : Différence ----------
    fig_diff = go.Figure(data=[go.Surface(z=phase_diff_np, colorscale='RdBu')])
    fig_diff.update_layout(
        title=f'Différence de Phase - Antenna {i}',
        scene=dict(
            xaxis_title='Doppler',
            yaxis_title='Range',
            zaxis_title='Différence de Phase (rad)'
        )
    )
    #fig_diff.show()
    fig_diff.write_html(f'./FNO/plot3D/phase_difference_antenna_{i}.html')

    fig_mag = go.Figure(data=[go.Surface(z=mag_diff_np, colorscale='RdBu')])
    fig_mag.update_layout(
        title=f'Différence de magnitude - Antenna {i}',
        scene=dict(
            xaxis_title='Doppler',
            yaxis_title='Range',
            zaxis_title='Différence de magnitude'
        )
    )
    #fig_diff.show()
    fig_mag.write_html(f'./FNO/plot3D/magnitude_difference_antenna_{i}.html')



