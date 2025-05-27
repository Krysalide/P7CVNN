import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

N = 32
batch = 10000

# Generate random input data and desired output data
sig = np.random.randn(batch, N) + 1j * np.random.randn(batch, N)
F = np.fft.fft(sig, axis=-1)

# First half of inputs/outputs is real part, second half is imaginary part
X = np.hstack([sig.real, sig.imag])
Y = np.hstack([F.real, F.imag])

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Define the model
class DFTModel(nn.Module):
    def __init__(self):
        super(DFTModel, self).__init__()
        self.linear = nn.Linear(N * 2, N * 2, bias=False)

    def forward(self, x):
        return self.linear(x)

model = DFTModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()

# Confirm that it works
def ANN_DFT(x):
    if len(x) != N:
        raise ValueError(f'Input must be length {N}')
    x_tensor = torch.tensor(np.hstack([x.real, x.imag]), dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_tensor)
    result = pred[:N] + 1j * pred[N:]
    return result

data = np.arange(N)
ANN = ANN_DFT(data)
FFT = np.fft.fft(data)

print(f'ANN matches FFT: {np.allclose(ANN, FFT)}')

# Heat map of neuron weights
plt.imshow(model.linear.weight.detach().numpy(), vmin=-1, vmax=1, cmap='coolwarm')
plt.show()
