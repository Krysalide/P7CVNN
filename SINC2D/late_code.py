import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinc(x):
    """1D sinc function."""
    return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(np.pi * x) / (np.pi * x))

def h_K_fs(k, fl, b, K, fs):
    """1D sinc filter."""
    term1 = 2 * (fl + b) * sinc(2 * (fl + b) * (k - b * K / 2) / fs)
    term2 = 2 * fl * sinc(2 * fl * (k - b * K / 2) / fs)
    return term1 - term2

def sinc_2D(n, m, fl_st, b_st, fl_ft, b_ft, N, M, fs_st, fs_ft):
    """2D sinc filter."""
    w = torch.outer(torch.cos(torch.linspace(-np.pi/2, np.pi/2, N, device=n.device)),
                    torch.cos(torch.linspace(-np.pi/2, np.pi/2, M, device=m.device)))
    h_st = h_K_fs(n, fl_st, b_st, N, fs_st)
    h_ft = h_K_fs(m, fl_ft, b_ft, M, fs_ft)
    return w * torch.outer(h_st, h_ft)

class SincConv2D(nn.Module):
    def __init__(self, filter_lengths, num_filters, sampling_frequencies):
        super(SincConv2D, self).__init__()
        self.filter_lengths = filter_lengths
        self.num_filters = num_filters
        self.sampling_frequencies = sampling_frequencies

        # Initialize filter parameters
        self.fl_st = nn.Parameter(torch.rand(num_filters))
        self.b_st = nn.Parameter(torch.rand(num_filters))
        self.fl_ft = nn.Parameter(torch.rand(num_filters))
        self.b_ft = nn.Parameter(torch.rand(num_filters))

    def forward(self, x):
        # Create 2D sinc filters
        n = torch.arange(self.filter_lengths[0], device=x.device)
        m = torch.arange(self.filter_lengths[1], device=x.device)
        filters = torch.zeros((self.num_filters, self.filter_lengths[0], self.filter_lengths[1]), device=x.device)

        for i in range(self.num_filters):
            filters[i, :, :] = sinc_2D(n, m,
                                       self.fl_st[i], self.b_st[i],
                                       self.fl_ft[i], self.b_ft[i],
                                       self.filter_lengths[0], self.filter_lengths[1],
                                       self.sampling_frequencies[0], self.sampling_frequencies[1])

        # Reshape filters for convolution operation
        filters = filters.unsqueeze(1)  # Add channel dimension

        # Apply filters to inputs
        return F.conv2d(x, filters, padding='same')

# Example usage
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.sinc_conv = SincConv2D(filter_lengths=(65, 33), num_filters=64, sampling_frequencies=(1000, 1000))
        self.conv1 = nn.Conv2d(64, 50, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 8 * 2, 32)  # Adjust input features based on your input size
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.sinc_conv(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 50 * 8 * 2)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model instance
model = CNN()
print(model)

# Example input tensor
input_tensor = torch.randn(1, 16, 512, 256)  # Batch size 1, 1 channel, 64x64 image
output = model(input_tensor)
print(output.shape)
