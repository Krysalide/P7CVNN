import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import matplotlib.pyplot as plt

from complexPyTorch.complexLayers import ComplexConv2d, ComplexReLU, ComplexBatchNorm2d, ComplexConvTranspose2d
from complexPyTorch.complexLayers import ComplexLinear
# to be tested
from activation_layers import CReLU, CPReLU, Naive_ComplexSigmoid, Naive_ComplexTanh,Cardioid


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
    name="ComplexUNet"
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(ComplexUNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = ComplexMaxPool2d(kernel_size=2, stride=2)  

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
                raise ValueError(f"Shape mismatch: {x.shape} vs {skip_connection.shape}")
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    


class ComplexOneLayer(nn.Module):
    name="linear_model"
    def __init__(self, in_channels, out_channels):
        super(ComplexOneLayer, self).__init__()
        
        self.linear_complex=ComplexLinear(in_features=in_channels,out_features=out_channels)

    

    def forward(self, x):
        

        return self.linear_complex(x)
    
class SmallComplexUNet(nn.Module):
    name="SmallComplexUNet"
    def __init__(self, in_channels, out_channels, features=[16,32,64,128]):
        super(SmallComplexUNet, self).__init__()
        
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
                raise ValueError(f"Shape mismatch: {x.shape} vs {skip_connection.shape}")
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    

class TinyComplexUNet(nn.Module):
    name="TinyComplexUNet"
    def __init__(self, in_channels, out_channels, features=[4,8,16,32]):
        super(TinyComplexUNet, self).__init__()
        
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
                raise ValueError(f"Shape mismatch: {x.shape} vs {skip_connection.shape}")
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


# apply_complex_custom is similar to the standard complex multiplication formula:
# (a + ib)(c + id) = (ac - bd) + i(ad + bc)

def apply_complex_custom(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


# light weight cnn, ram memory friendly
class ComplexLinearNoBias(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinearNoBias, self).__init__()
        self.fc_r = Linear(in_features, out_features,bias=False)
        self.fc_i = Linear(in_features, out_features,bias=False)

    def forward(self, input):
        return apply_complex_custom(self.fc_r, self.fc_i, input)

def get_complex_weights(layer: ComplexLinearNoBias):
    """Retourne les poids complexes d'une couche ComplexLinearNoBias."""
    
    if not isinstance(layer, ComplexLinearNoBias):
        raise TypeError("La couche fournie n'est pas une instance de ComplexLinearNoBias.")
    
    # Récupère les poids réels et imaginaires
    weight_r = layer.fc_r.weight.data
    weight_i = layer.fc_i.weight.data

    # Combine pour créer un tenseur complexe
    weight_complex = weight_r + 1j * weight_i
    return weight_complex

def visualize_complex_plane(layer: ComplexLinearNoBias):
    weight_r = layer.fc_r.weight.data.cpu().numpy()
    weight_i = layer.fc_i.weight.data.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(weight_r.flatten(), weight_i.flatten(), alpha=0.7)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel("Partie réelle")
    plt.ylabel("Partie imaginaire")
    plt.title("Poids complexes du modèle complexe lineaire")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_complex_norm(layer: ComplexLinearNoBias,interactive=False):
    weight_r = layer.fc_r.weight.data.cpu().numpy()
    weight_i = layer.fc_i.weight.data.cpu().numpy()
    norm = (weight_r ** 2 + weight_i ** 2) ** 0.5

    plt.figure(figsize=(6, 4))
    plt.imshow(norm, cmap='magma')
    plt.title("Norme des poids complexes")
    plt.xlabel("Entrées")
    plt.ylabel("Sorties")
    plt.colorbar(label="Norme")
    plt.tight_layout()
    if interactive:
        plt.show()
        return 'dummy_path'
    else:
        save_path='/home/christophe/ComplexNet/FFT/complex_norm.png'
        plt.savefig(save_path)
        return save_path


def visualize_complex_weights(layer: ComplexLinearNoBias,interactive=False):
    weight_r = layer.fc_r.weight.data.cpu().numpy()
    weight_i = layer.fc_i.weight.data.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(weight_r, cmap='viridis')
    axs[0].set_title("Poids réels")
    axs[0].set_xlabel("Entrées")
    axs[0].set_ylabel("Sorties")

    axs[1].imshow(weight_i, cmap='plasma')
    axs[1].set_title("Poids imaginaires")
    axs[1].set_xlabel("Entrées")
    axs[1].set_ylabel("Sorties")
    plt.tight_layout()
    if interactive:
        plt.show()
        return 'dummy_path'
    else:
        save_path='/home/christophe/ComplexNet/FFT/complex_weights.png'
        plt.savefig(save_path)
        plt.close()
        return save_path



def complex_mse_loss(output, target):
    output_re = output.real
    output_im = output.imag
    target_re = target.real
    target_im = target.imag
    loss_re = F.mse_loss(output_re, target_re)
    loss_im = F.mse_loss(output_im, target_im)
    return loss_re + loss_im

def phase_loss(output, target):
    
    phase_diff = torch.angle(output) - torch.angle(target)
    
    #phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    return torch.mean(torch.abs(phase_diff))

def hybrid_loss(output, target):
    mse_loss = complex_mse_loss(output, target)
    phase_loss_value = phase_loss(output, target)
    return mse_loss * phase_loss_value

def is_empirically_convex(loss_fn, x1, x2, target, device='cpu', samples=10):
    lambdas = torch.linspace(0, 1, samples)
    convex_violations = 0

    for lam in lambdas:
        x_interp = lam * x1 + (1 - lam) * x2
        lhs = loss_fn(x_interp, target)
        rhs = lam * loss_fn(x1, target) + (1 - lam) * loss_fn(x2, target)

        if lhs > rhs + 1e-6:  
            convex_violations += 1

    return convex_violations == 0 




# allows to test the model without using the dataloader
# and the training loop
if __name__ == '__main__':
    
    batch_size = 2
    in_channels = 16  
    out_channels = 16  
    height, width = 256, 512 # radar image size

    
    real_part = torch.randn(batch_size, in_channels, height, width).multiply(100)
    imag_part = torch.randn(batch_size, in_channels, height, width).multiply(100)
    complex_input = torch.complex(real_part, imag_part)

    #model = SmallComplexUNet(in_channels=in_channels, out_channels=out_channels)

    model=TinyComplexUNet(in_channels=in_channels, out_channels=out_channels)
    print(model.name)

    
    #model = ComplexUNet(in_channels=in_channels, out_channels=out_channels)

    model.eval()
    with torch.no_grad():
        output = model(complex_input)

        
        print("Forme de l'entrée:", complex_input.shape)
        print("Forme de la sortie:", output.shape)
        loss = complex_mse_loss(output, complex_input)
        print("MSE loss:", loss.item())
        loss=phase_loss(complex_input, output)
        print("Phase loss:", loss.item())
        print("diff between phase loss an pi/2:", torch.abs(loss - 3.14159/2))

    device='cuda'
    x1 = torch.randn(1, 1, 64, 64, dtype=torch.complex64, requires_grad=True).to(device)
    x2 = torch.randn(1, 1, 64, 64, dtype=torch.complex64, requires_grad=True).to(device)
    target = torch.randn(1, 1, 64, 64, dtype=torch.complex64).to(device)
    is_convex=is_empirically_convex(loss_fn=complex_mse_loss,x1=x1,x2=x2,target=target,device=device,samples=1000)
    print('convexity of complex mse loss: ',is_convex)
    is_convex=is_empirically_convex(loss_fn=hybrid_loss,x1=x1,x2=x2,target=target,device=device,samples=1000)
    print('convexity of hybrid loss: ',is_convex)

    is_convex=is_empirically_convex(loss_fn=phase_loss,x1=x1,x2=x2,target=target,device=device,samples=1000)
    print('convexity of phase loss: ',is_convex)
