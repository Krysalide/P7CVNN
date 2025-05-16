import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ??? might be removed
class RadarDatasetCuda(Dataset):
    def __init__(self, save_folder, indices, device='cuda', transform=None, target_transform=None):
        self.save_folder = save_folder
        self.indices = indices
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Load complex-valued numpy arrays
        x = np.load(os.path.join(self.save_folder, f'raw_adc_{i}.npy'))  # (512, 256, 16), complex64
        y = np.load(os.path.join(self.save_folder, f'range_doppler_map_{i}.npy'))  # could also be complex

        # Convert to torch tensors
        x = torch.from_numpy(x).to(self.device)  # torch.complex64 on GPU
        y = torch.from_numpy(y).to(self.device)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

class RadarDataset(Dataset):
    def __init__(self, save_folder, indices, transform=None, target_transform=None):
        self.save_folder = save_folder
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        
        x = np.load(os.path.join(self.save_folder, f'raw_adc_{i}.npy'))   # shape: (512, 256, 16), dtype: complex64
        y = np.load(os.path.join(self.save_folder, f'range_doppler_map_{i}.npy'))  # shape: ?, dtype: complex64 or float

        
        x = torch.from_numpy(x)  
        y = torch.from_numpy(y)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

def load_data(save_folder, indices):
    dataset = RadarDataset(save_folder, indices)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    return dataloader

def main():
    sequence = 'RECORD@2020-11-21_11.54.31'
    save_folder = f'/media/christophe/backup/DATARADIAL/{sequence}'

    
    indices = list(range(250))

    dataset = RadarDataset(save_folder, indices)
    print(f"Dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    print(f"DataLoader length: {len(dataloader)}")
    for x, y in dataloader:
        print(x.shape, x.dtype)  # e.g., torch.Size([8, 512, 256, 16]) torch.complex64
        print(y.shape,y.dtype)
        break
    

if __name__ == "__main__":
    main()