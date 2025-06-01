import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


''''
To be kept for presentation ??
shows weights after training, good to compare between initial and final
seems to show it is better to initialize weights with real DFT
although we have good loss results
It is possible to initiate weights with DFT or randomly
neural network able to learn 1D DFT very efficiently



'''

class LearnableDFT(nn.Module):
    def __init__(self, N, learnable=True):
        super().__init__()
        self.N = N
        t = np.arange(N)
        k = t.reshape((N, 1))

        #Matrices DFT "vraies"
        F_re = np.cos(2 * np.pi * k * t / N)
        F_im = -np.sin(2 * np.pi * k * t / N)

        # Matrices DFT aléatoires (pour voir si le réseau apprend)
        # F_re = np.random.randn(N, N)  
        # F_im = np.random.randn(N, N)

        self.linear_re = nn.Linear(N, N, bias=False)
        self.linear_im = nn.Linear(N, N, bias=False)

        self.linear_re.weight = nn.Parameter(torch.tensor(F_re, dtype=torch.float32), requires_grad=learnable)
        self.linear_im.weight = nn.Parameter(torch.tensor(F_im, dtype=torch.float32), requires_grad=learnable)

    def forward(self, x):
        X_re = self.linear_re(x)
        X_im = self.linear_im(x)
        return X_re, X_im  
    
N = 512
batch_size = 128
num_epochs = 3500
lr = 0.01


X = torch.randn(batch_size, N)

# Cible : FFT de référence (parties réelle et imaginaire)
X_fft = torch.fft.fft(X)
target_re = X_fft.real
target_im = X_fft.imag


model = LearnableDFT(N, learnable=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr,differentiable=False)
loss_fn = nn.MSELoss()

# Boucle d'entraînement
losses = []
with torch.no_grad():
    W_re_initial = model.linear_re.weight.detach().numpy()
    W_im_initial= model.linear_im.weight.detach().numpy()


for epoch in range(num_epochs):
    pred_re, pred_im = model(X)
    loss = loss_fn(pred_re, target_re) + loss_fn(pred_im, target_im)
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

plt.plot(losses)
plt.title("Évolution de la perte")
plt.xlabel("Époque")
plt.ylabel("Loss MSE")
plt.grid(True)
plt.show()

with torch.no_grad():
    W_re = model.linear_re.weight.detach().numpy()
    W_im = model.linear_im.weight.detach().numpy()

fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].imshow(W_re, aspect='auto', cmap='coolwarm')
axs[0, 0].set_title("Poids Réels appris")

axs[0, 1].imshow(W_im, aspect='auto', cmap='coolwarm')
axs[0, 1].set_title("Poids Imaginaires appris")

# FFT computed
F_re = np.cos(2 * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
F_im = -np.sin(2 * np.pi * np.outer(np.arange(N), np.arange(N)) / N)

axs[1, 0].imshow(F_re, aspect='auto', cmap='coolwarm')
axs[1, 0].set_title("Ground truth real part (DFT)")

axs[1, 1].imshow(F_im, aspect='auto', cmap='coolwarm')
axs[1, 1].set_title("Ground truth imaginary part (DFT)")

plt.tight_layout()
plt.show()




fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].imshow(W_re_initial, aspect='auto', cmap='coolwarm')
axs[0, 0].set_title("Poids Réels initiaux")

axs[0, 1].imshow(W_im_initial, aspect='auto', cmap='coolwarm')
axs[0, 1].set_title("Poids Imaginaires initiaux")

# FFT computed
F_re = np.cos(2 * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
F_im = -np.sin(2 * np.pi * np.outer(np.arange(N), np.arange(N)) / N)

axs[1, 0].imshow(F_re, aspect='auto', cmap='coolwarm')
axs[1, 0].set_title("Ground truth (DFT)")

axs[1, 1].imshow(F_im, aspect='auto', cmap='coolwarm')
axs[1, 1].set_title("Ground truth (DFT)")

plt.tight_layout()
plt.show()
