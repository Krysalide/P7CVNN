import torch
import torch.nn as nn
import torch.nn.functional as F

from complexPyTorch.complexLayers import ComplexConv2d, ComplexReLU, ComplexBatchNorm2d, ComplexConvTranspose2d

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.real_pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.imag_pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        real_part = torch.real(x)
        imag_part = torch.imag(x)
        pooled_real = self.real_pool(real_part)
        pooled_imag = self.imag_pool(imag_part)
        return torch.complex(pooled_real, pooled_imag)

class ComplexUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(ComplexUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = ComplexMaxPool2d(kernel_size=2, stride=2)  # Utilisez la couche de pooling complexe

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(ComplexDoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ComplexDoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                ComplexConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ComplexDoubleConv(feature * 2, feature))

        self.final_conv = ComplexConv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

class ComplexDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(),
            ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU()
        )

    def forward(self, x):
        return self.conv(x)

def complex_mse_loss(output_re, output_im, target_re, target_im):
    loss_re = F.mse_loss(output_re, target_re)
    loss_im = F.mse_loss(output_im, target_im)
    return loss_re + loss_im

def phase_loss(output, target):
    # Calculer la perte de phase
    phase_diff = torch.angle(output) - torch.angle(target)
    # Ramener l'erreur entre -π et π
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    return torch.mean(torch.abs(phase_diff))

if __name__ == '__main__':
    # Exemple d'utilisation:
    batch_size = 2
    in_channels = 16  
    out_channels = 16  
    height, width = 256, 512 # radar image size

    # Créer un tenseur d'entrée complexe aléatoire
    real_part = torch.randn(batch_size, in_channels, height, width)
    imag_part = torch.randn(batch_size, in_channels, height, width)
    complex_input = torch.complex(real_part, imag_part)

    # Instancier le modèle ComplexUNet
    model = ComplexUNet(in_channels=in_channels, out_channels=out_channels)

    # Effectuer une passe avant
    output = model(complex_input)

    # Afficher la forme de la sortie
    print("Forme de l'entrée:", complex_input.shape)
    print("Forme de la sortie:", output.shape)

    loss=phase_loss(complex_input, output)
    print("Phase loss:", loss.item())