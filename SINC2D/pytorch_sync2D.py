import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fs_st, fs_ft, Nst, Nft,
                 stride=(1, 1), padding='valid'):
        super().__init__()
        self.in_channels = in_channels
        self.N, self.M = kernel_size
        self.stride = stride
        self.padding = padding

        self.fs_st = fs_st
        self.fs_ft = fs_ft

        # Trainables
        self.fl_st = nn.Parameter(torch.linspace(50, 150, Nst).view(-1, 1))  # (Nst, 1)
        self.bw_st = nn.Parameter(torch.full((Nst, 1), 50.))
        self.fl_ft = nn.Parameter(torch.linspace(100, 300, Nft).view(1, -1))  # (1, Nft)
        self.bw_ft = nn.Parameter(torch.full((1, Nft), 100.))

        self.Nst = Nst
        self.Nft = Nft
        self.out_channels = Nst * Nft

        # Grilles n, m
        n = torch.arange(self.N) - (self.N // 2)
        m = torch.arange(self.M) - (self.M // 2)
        self.register_buffer('n', n)
        self.register_buffer('m', m)

        # Fenêtre cosinus 2D
        win_n = 0.5 - 0.5 * torch.cos(2 * np.pi * torch.arange(self.N) / (self.N - 1))
        win_m = 0.5 - 0.5 * torch.cos(2 * np.pi * torch.arange(self.M) / (self.M - 1))
        window_2d = torch.outer(win_n, win_m)
        self.register_buffer('window', window_2d)

    def forward(self, x):
        n = self.n.view(1, self.N, 1)  # (1, N, 1)
        m = self.m.view(1, 1, self.M)  # (1, 1, M)

        # -- Slow-time (Nst, N, 1)
        f1_st = self.fl_st  # (Nst, 1)
        f2_st = f1_st + torch.abs(self.bw_st)
        f1_st = f1_st.view(-1, 1, 1)  # (Nst, 1, 1)
        f2_st = f2_st.view(-1, 1, 1)

        h_n = (2 * f2_st / self.fs_st) * torch.sinc(2 * f2_st * n / self.fs_st) - \
          (2 * f1_st / self.fs_st) * torch.sinc(2 * f1_st * n / self.fs_st)  # (Nst, N, 1)

        # -- Fast-time (1, Mft, M)
        f1_ft = self.fl_ft  # (1, Nft)
        f2_ft = f1_ft + torch.abs(self.bw_ft)
        f1_ft = f1_ft.view(1, -1, 1)  # (1, Nft, 1)
        f2_ft = f2_ft.view(1, -1, 1)

        h_m = (2 * f2_ft / self.fs_ft) * torch.sinc(2 * f2_ft * m / self.fs_ft) - \
          (2 * f1_ft / self.fs_ft) * torch.sinc(2 * f1_ft * m / self.fs_ft)  # (1, Nft, M)

        # -- Produit tensoriel : (Nst, Nft, N, M)
        filters = h_n @ h_m  # (Nst, N, 1) @ (1, Nft, M) → (Nst, Nft, N, M)

        filters = filters.permute(0, 1, 2, 3)  # (Nst, Nft, N, M)
        filters = filters.reshape(self.out_channels, 1, self.N, self.M)
        filters = filters * self.window  # fenêtrage

        # Répliquer les filtres sur tous les canaux d'entrée
        filters = filters.repeat(1, self.in_channels, 1, 1)  # (out, in, N, M)

        if self.padding == 'same':
            pad = (self.M // 2, self.M // 2, self.N // 2, self.N // 2)
            x = F.pad(x, pad, mode='reflect')

        return F.conv2d(x, filters, stride=self.stride, padding=0)

x = torch.randn(8, 16, 512, 256)  # batch=8, 16 canaux, slow=512, fast=256

sinc_layer = SincConv2D(
    in_channels=16,
    out_channels=64,         # Nst*Nft
    kernel_size=(65, 33),
    fs_st=2000, fs_ft=2000,
    Nst=8, Nft=8,
    stride=(4, 4),
    padding='same'
)

y = sinc_layer(x)
print(y.shape)  # (8, 64, ..., ...)
