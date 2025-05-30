import numpy as np
import time 
import torch
import torch.nn as nn
import torch.optim as optim

''' 
New approach with data shape compatible with valeo radar data
warning see amplitude? 
With 100 samples no better loss


'''

# Parameters
H, W = 512, 256
antenna = 16
patch_size = 32
n_channels = 2  # real + imag
patch_dim = n_channels * patch_size * patch_size
n_samples = 100  # Number of samples seems limited to 200 in memory so it would be nice to use a dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('device: ',device)
assert device == 'cuda' 
# https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html
unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)).to(device)

# Prepare dataset
X_all = []
Y_all = []

for _ in range(n_samples):
    sig = np.random.randn(antenna, H, W) + 1j * np.random.randn(antenna, H, W)
    F = np.fft.fft2(sig, axes=(-2, -1))

    sig_np = np.stack([sig.real, sig.imag], axis=1)
    F_np = np.stack([F.real, F.imag], axis=1)

    X_tensor = torch.tensor(sig_np, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(F_np, dtype=torch.float32).to(device)

    X_patches = unfold(X_tensor).transpose(1, 2).reshape(-1, patch_dim)  # (batch * n_patches, patch_dim)
    Y_patches = unfold(Y_tensor).transpose(1, 2).reshape(-1, patch_dim)

    X_all.append(X_patches)
    Y_all.append(Y_patches)

# Concatenate all samples
X_data = torch.cat(X_all, dim=0)
Y_data = torch.cat(Y_all, dim=0)

print(f"Total patches: {X_data.shape[0]}")  

# Model
resume_training = True
PATH = '/home/christophe/ComplexNet/FFT/fft_patch_lr3.pth'
model = nn.Linear(patch_dim, patch_dim, bias=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.9)

if resume_training:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 # not used yet
else:
    print('Training will start from scratch')


loss_fn = nn.MSELoss()

# Training loop
n_epochs = 200 # pretty fast
batch_size = 100
start_time=time.time()
for epoch in range(n_epochs):
    
    model.train()

    perm = torch.randperm(X_data.size(0))
    X_shuffled = X_data[perm]
    Y_shuffled = Y_data[perm]

    for i in range(0, X_data.size(0), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")


save_path='/home/christophe/ComplexNet/FFT/fft_patch_lr3.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, save_path)

print('model saved succesfully')


