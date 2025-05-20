import numpy as np
import time
import os
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from data_reader import RadarDataset, split_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset ---
sequence = 'RECORD@2020-11-21_11.54.31'
save_folder = f'/media/christophe/backup/DATARADIAL/{sequence}'
dataset = RadarDataset(save_folder, list(range(250)))

train_loader, val_loader, test_loader = split_dataloader(dataset)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")


# --- Utility Functions ---
def split_real_imag_torch(X):
    return torch.cat([X.real, X.imag], dim=1)  # shape: (B, 32, H, W)

def downsample_torch_gpu(X, out_size=(64, 32)):
    return F.interpolate(X, size=out_size, mode='bilinear', align_corners=False)

def prepare_data_gpu(loader):
    X_all, Y_all = [], []
    #start_time = time.time()
    count=0
    for X, Y in loader:
        count+=1
        # Move to device
        X = X.to(device).to(torch.cfloat)  # assume complex64
        Y = Y.to(device).to(torch.cfloat)

        # Split and downsample
        X = split_real_imag_torch(X)
        Y = split_real_imag_torch(Y)
        #print("Data first step time:", time.time() - start_time)
        #start_time = time.time()

        X = downsample_torch_gpu(X, out_size=(64, 32))
        Y = downsample_torch_gpu(Y, out_size=(64, 32))
        #print("Data second step time:", time.time() - start_time)
        # Flatten
        #start_time = time.time()
        X = X.view(X.size(0), -1).cpu()  # move back to CPU
        Y = Y.view(Y.size(0), -1).cpu()
        #print("Data third step time:", time.time() - start_time)
        #start_time = time.time()
        X_all.append(X)
        Y_all.append(Y)
        print(count)
    #print("Data fourth step time:", time.time() - start_time)

    print('Count: ',count)
    return torch.cat(X_all, dim=0).numpy(), torch.cat(Y_all, dim=0).numpy()

# --- Prepare train and test data ---
X_train, Y_train = prepare_data_gpu(train_loader)
X_test, Y_test = prepare_data_gpu(test_loader)
print('data preparation done')
# --- PCA + Ridge ---
n_components = 100
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print('PCA done')
reg = Ridge()
reg.fit(X_train_pca, Y_train)
Y_pred_flat = reg.predict(X_test_pca)

# --- Evaluate ---
mse = mean_squared_error(Y_test, Y_pred_flat)
print("Test MSE:", mse)

# --- Optional: reshape prediction ---
#Y_pred_down = Y_pred_flat.reshape(Y_pred_flat.shape[0], 32, 64, 32)
Y_pred_down = Y_pred_flat.reshape(Y_test.shape[0], 32, 64, 32)
print("Prediction shape:", Y_pred_down.shape)
