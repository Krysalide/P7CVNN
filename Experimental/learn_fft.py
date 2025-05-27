import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 32
batch = 10000

# Generate training data
sig = np.random.randn(batch, N) + 1j * np.random.randn(batch, N)
F = np.fft.fft(sig, axis=-1)

# Stack real and imaginary parts
X = np.hstack([sig.real, sig.imag]).astype(np.float32)
Y = np.hstack([F.real, F.imag]).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Define the model: one linear layer, no bias
model = nn.Linear(N * 2, N * 2, bias=False)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 300
batch_size = 100

for epoch in range(epochs):
    print(epoch)
    permutation = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_tensor[indices], Y_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

print('training done')
# Confirm that it works
def ANN_DFT(x_np):
    if len(x_np) != N:
        raise ValueError(f'Input must be length {N}')
    x_input = np.hstack([x_np.real, x_np.imag]).astype(np.float32)
    x_tensor = torch.tensor(x_input).unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x_tensor).squeeze().numpy()
    return y_pred[:N] + 1j * y_pred[N:]

data = np.arange(N)
ANN = ANN_DFT(data)
FFT = np.fft.fft(data)
print(f'ANN matches FFT: {np.allclose(ANN, FFT, atol=1e-1)}')  # allow slight tolerance

# Plot the learned weights
weights = model.weight.detach().numpy()
plt.imshow(weights, vmin=-1, vmax=1, cmap='coolwarm')
plt.title("Learned DFT Weights")
plt.colorbar()
plt.show()


