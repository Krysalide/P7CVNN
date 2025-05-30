import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from data_fft_reader import RadarFFTDataset

from data_fft_reader import split_dataloader

''''
Inspired from 1D fft layer
Model consist of one Linear layer
Structure not adapted to our size of radar data
Would be cool to keep the plots of weights

Memory too large with our radar shape!!!

Can be  deleted
'''

# Parameters
H, W = 32, 32 # height and width of 2D signal
N = H * W
batch = 16

#  random complex 2D signals
sig = np.random.randn(batch, H, W) + 1j * np.random.randn(batch, H, W)
F = np.fft.fft2(sig, axes=(-2, -1))

# Prepare training data: flatten and stack real + imaginary parts
X = np.concatenate([sig.real.reshape(batch, -1), sig.imag.reshape(batch, -1)], axis=1).astype(np.float32)
Y = np.concatenate([F.real.reshape(batch, -1), F.imag.reshape(batch, -1)], axis=1).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Define the model: one big linear layer
model = nn.Linear(N * 2, N * 2, bias=False)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
epochs = 20000
batch_size = 10

for epoch in range(epochs):
    permutation = torch.randperm(X_tensor.size(0))
    if epoch%100==0:
        print(epoch)
    for i in range(0, X_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_tensor[indices], Y_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        #print(loss.item())
        loss.backward()
        optimizer.step()


print('last loss: ',loss.item())


# Function to predict 2D FFT using trained model
def ANN_2DFFT(x_np):
    if x_np.shape != (H, W):
        raise ValueError(f'Input must be shape ({H}, {W})')
    x_input = np.concatenate([x_np.real.flatten(), x_np.imag.flatten()]).astype(np.float32)
    x_tensor = torch.tensor(x_input).unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x_tensor).squeeze().numpy()
    real_part = y_pred[:N].reshape(H, W)
    imag_part = y_pred[N:].reshape(H, W)
    return real_part + 1j * imag_part

# Test on simple input
data = np.outer(np.arange(H), np.arange(W))  # a structured test pattern
ANN = ANN_2DFFT(data)
FFT = np.fft.fft2(data)
print(f'ANN matches FFT: {np.allclose(ANN, FFT, atol=1e-1)}')

# Visualize learned weights
weights = model.weight.detach().numpy()
plt.imshow(weights, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Learned 2D DFT Weights")
plt.colorbar()
save_path='/home/christophe/ComplexNet/Experimental/weights16_16.png'
plt.savefig(save_path)
plt.show()
