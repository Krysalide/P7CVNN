import time
import numpy as np
import torch
import torch.nn as nn
from ComplexUnet import ComplexUNet
from radar_metrics import complex_mse_per_antenna, complex_mae_per_antenna, phase_error_per_antenna, relative_error_per_antenna, real_imag_mse_per_antenna
from p7_utils import list_record_folders,create_dataloaders
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
# conda env: complex_net

PATH='/home/christophe/ComplexNet/complex_net1.pth'
model = ComplexUNet(in_channels=16, out_channels=16).to(device)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()


def print_model_layers(model: nn.Module):
    print("=== Architecture du modèle ===")
    count_layers = 0
    for name, module in model.named_modules():
        if name == "":
            # Ignorer le module racine (le modèle lui-même)
            continue
        count_layers += 1
        
        print(f"{name}: {module.__class__.__name__}")
    print("===============================")
    print(f"Total number of layers: {count_layers}")
    print("=== Fin de l'architecture du modèle ===")
print_model_layers(model)
folders_adc=list_record_folders('/media/christophe/backup/DATASET/ADC/')
#print(folders_adc)
folders_range_doppler=list_record_folders('/media/christophe/backup/DATASET/RDGD/')

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

train_loader, val_loader, test_loader = create_dataloaders(
        adc_data,
        rd_data, 
        batch_size=2,
        test_split=0.9, 
    )
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
with torch.no_grad():
    for batch_data, batch_target in test_loader:
        x = batch_data.permute(0, 3, 1, 2).to(device, torch.complex64)
        y = batch_target.permute(0, 3, 1, 2).to(device, torch.complex64)

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