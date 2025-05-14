import torch
from torch import nn
import torch.nn.functional as F

# code found on github
#https://github.com/saurabhya/FCCNs/blob/main/complex_activations.py
# PRETTY SIMPLE COMPLEX ACTIVATIONS
# to be tested
# see also cardioid activation function
class CReLU(nn.Module):
    ''''
    Simply apply ReLU to the real and imaginary parts of the complex number
    and concatenate the results.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x.real).type(torch.complex64) + 1j * F.relu(x.imag).type(torch.complex64)

class CPReLU(nn.Module):
    def __init__(self, num_channels= 1):
        super().__init__()
        self.num_channels= num_channels
        self.real_prelu = nn.PReLU(num_parameters= self.num_channels)
        self.imag_prelu = nn.PReLU(num_parameters= self.num_channels)

    def forward(self, x):
        return self.real_prelu(x.real).type(torch.complex64) + 1j * self.imag_prelu(x.imag).type(torch.complex64)

class zReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        real_mask=x.real>0
        imag_mask=x.imag>0
        mask = real_mask & imag_mask 
        return x * mask
         

class Naive_ComplexSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x.real).type(torch.complex64) + 1j * F.sigmoid(x.imag).type(torch.complex64)

class Naive_ComplexTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.tanh(x.real).type(torch.complex64) + 1j * F.tanh(x.imag).type(torch.complex64)

class ModReLU(nn.Module):
    def __init__(self, bias):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, z: torch.Tensor):
        # z is a complex tensor
        abs_z = torch.abs(z)
        phase = z / (abs_z + 1e-6)  # Avoid division by zero
        activated = torch.relu(abs_z + self.bias)
        return activated * phase

class Cardioid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor):
        
        abs_z = torch.abs(z)
        real_z = z.real
        
        cos_theta = real_z / (abs_z + 1e-6)
        factor = 0.5 * (1 + cos_theta)
        return z * factor


class cardioid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x.real * torch.cos(x.imag) - x.imag * torch.sin(x.real)).type(torch.complex64) + \
               1j * (x.real * torch.sin(x.imag) + x.imag * torch.cos(x.real)).type(torch.complex64)
    
if __name__ == '__main__':
    pred = torch.randn(2, 16, 256, 512, dtype=torch.cfloat)
    cardioid = Cardioid()
    output = cardioid(pred)
    assert output.shape == pred.shape
    print("Cardioid shape:", output.shape)
    crelu = CReLU()
    output = crelu(pred)
    assert output.shape == pred.shape
    mod_relu= ModReLU(bias=0.5)
    output = mod_relu(pred)
    assert output.shape == pred.shape
    print("ModRelu Output shape:", output.shape)
    crelu = CReLU()
    output = crelu(pred)
    assert output.shape == pred.shape
    print("CReLU Output shape:", output.shape)
    prelu = CPReLU(num_channels=16)
    output = prelu(pred)
    assert output.shape == pred.shape
    print("PReLU Output shape:", output.shape)
    zrelu = zReLU()
    output = zrelu(pred)
    assert output.shape == pred.shape
    print("zReLU Output shape:", output.shape)
    sigmoid = Naive_ComplexSigmoid()
    output = sigmoid(pred)
    assert output.shape == pred.shape
    print("Naive Complex Sigmoid Output shape:", output.shape)
    tanh = Naive_ComplexTanh()
    output = tanh(pred)
    assert output.shape == pred.shape
    print("Naive Complex Tanh Output shape:", output.shape)
    
