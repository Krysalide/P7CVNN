import torch
from ComplexUnet import complex_mse_loss,hybrid_loss
from ComplexUnet import phase_loss,is_empirically_convex
from neuralop.models import FNO2d
from neuralop.losses import H1Loss
from p7_utils import print_model_layers

if __name__ == '__main__':

    print('must be used with fno_env!!')  
    print('Pytorch version: ',torch.__version__)  
    batch_size = 2
    in_channels = 16  
    out_channels = 16  
    height, width = 256, 512 # radar image size

    
    real_part = torch.randn(batch_size, in_channels, height, width).multiply(100)
    imag_part = torch.randn(batch_size, in_channels, height, width).multiply(100)
    complex_input = torch.complex(real_part, imag_part)
    model=FNO2d(complex_data=True,in_channels=16,out_channels=16,n_modes_height=256,n_modes_width=512,hidden_channels=16)
    print(50*'####')
    print(model)
    print(50*"-")
    print_model_layers(model)
    
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


