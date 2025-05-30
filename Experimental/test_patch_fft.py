import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

H, W = 256, 512
N = H * W
batch = 16

# Patch parameters
patch_size = 32
n_channels = 2  # real + imag
patch_dim = n_channels * patch_size * patch_size

PATH='/home/christophe/ComplexNet/FFT/fft_patch.pth'
model = nn.Linear(patch_dim, patch_dim, bias=False)
model.load_state_dict(torch.load(PATH, weights_only=True))

loss_fn = nn.MSELoss()

test_batch = 16
sig_test = np.random.randn(test_batch, H, W) + 1j * np.random.randn(test_batch, H, W)
F_test = np.fft.fft2(sig_test, axes=(-2, -1))

# Stack real/imag
sig_test_np = np.stack([sig_test.real, sig_test.imag], axis=1)
F_test_np   = np.stack([F_test.real, F_test.imag], axis=1)

# Convert to tensors
X_test_tensor = torch.tensor(sig_test_np, dtype=torch.float32)
Y_test_tensor = torch.tensor(F_test_np, dtype=torch.float32)

unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

# Unfold test patches
X_test_patches = unfold(X_test_tensor)
Y_test_patches = unfold(Y_test_tensor)

# Mise en forme
X_test_patches = X_test_patches.transpose(1, 2).reshape(-1, patch_dim)
Y_test_patches = Y_test_patches.transpose(1, 2).reshape(-1, patch_dim)

# Pas d'entraînement ici
model.eval()
with torch.no_grad():
    Y_test_pred = model(X_test_patches)

# Erreur MSE
mse = loss_fn(Y_test_pred, Y_test_patches).item()
print(f"MSE on test set: {mse:.6f}")


def psnr(mse, max_val=1.0):
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

print(f"PSNR: {psnr(mse):.2f} dB")


n_patches_per_image = (H // patch_size) * (W // patch_size)
Y_test_pred_reshaped = Y_test_pred.reshape(test_batch, n_patches_per_image, patch_dim).transpose(1, 2)

fold = nn.Fold(output_size=(H, W), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

# Replié (batch, 2*C, H, W)
Y_pred_folded = fold(Y_test_pred_reshaped)


# Pour normaliser (car Fold fait une somme des patches superposés)
ones = torch.ones_like(X_test_tensor)
ones_patches = unfold(ones).transpose(1, 2).reshape(test_batch, n_patches_per_image, patch_dim)
ones_folded = fold(ones_patches.transpose(1, 2))

# Éviter division par 0
Y_pred_final = Y_pred_folded / ones_folded


print("Shape finale reconstituée :", Y_pred_final.shape)
# Résultat attendu : (test_batch, 2, H, W)




# Afficher la magnitude du FFT prédite et réelle pour un exemple
idx = 0
true_mag = torch.sqrt(Y_test_tensor[idx, 0]**2 + Y_test_tensor[idx, 1]**2)
pred_mag = torch.sqrt(Y_pred_final[idx, 0]**2 + Y_pred_final[idx, 1]**2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(true_mag.numpy(), cmap='viridis')
plt.title("FFT réelle")
plt.subplot(1, 3, 2)
plt.imshow(pred_mag.numpy(), cmap='viridis')
plt.title("FFT prédite")
plt.subplot(1, 3, 3)
plt.imshow(pred_mag.numpy()-true_mag.numpy(), cmap='viridis')
plt.title("Diff")
plt.colorbar()
plt.show()






