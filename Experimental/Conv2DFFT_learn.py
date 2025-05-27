import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parameters
H, W = 16, 16
batch = 10000
in_channels = 2  # real + imag
out_channels = 2  # Number of "Fourier-like" filters to learn
kernel_size = (H, W)  # Full image filters for now

# Generate random complex images
sig = np.random.randn(batch, H, W) + 1j * np.random.randn(batch, H, W)
F = np.fft.fft2(sig)

# Input: [batch, 2, H, W]  → real + imag as separate channels
X_real = sig.real[:, np.newaxis, :, :]
X_imag = sig.imag[:, np.newaxis, :, :]
X = np.concatenate([X_real, X_imag], axis=1).astype(np.float32)

# Target: FFT magnitude
Y = np.abs(F).astype(np.float32)
Y = Y[:, np.newaxis, :, :]  # [batch, 1, H, W]

# Convert to tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Define model: 2D convolution layer
class ConvDFTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False
        )

    def forward(self, x):
        # Resulting shape: [batch, out_channels, 1, 1]
        x = self.conv(x)
        # Squeeze to [batch, out_channels]
        return x.view(x.size(0), -1)

model = ConvDFTNet()

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Training loop
epochs = 50
batch_size = 100

for epoch in range(epochs):
    perm = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        idx = perm[i:i + batch_size]
        batch_x = X_tensor[idx]
        #batch_y = Y_tensor[idx].view(batch_size, -1)  # Flatten
        batch_y = Y_tensor[idx].mean(dim=[2, 3])

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Examine learned filters
with torch.no_grad():
    weights = model.conv.weight.numpy()  # [out_channels, 2, H, W]

# Visualize the learned real and imaginary parts
fig, axs = plt.subplots(out_channels, 2, figsize=(6, out_channels * 2))
for i in range(out_channels):
    axs[i, 0].imshow(weights[i, 0], cmap='coolwarm')
    axs[i, 0].set_title(f'Filter {i} - Real')
    axs[i, 1].imshow(weights[i, 1], cmap='coolwarm')
    axs[i, 1].set_title(f'Filter {i} - Imag')
plt.tight_layout()
plt.show()
