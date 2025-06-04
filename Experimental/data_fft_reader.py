import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import  Subset
from glob import glob


'''
Clone of main dataset
Could be replaced by initial one 
'''

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
    
class RadarDatasetV2(Dataset):
    def __init__(self, save_folder, indices=None, transform=None, target_transform=None, recursive=False):
        self.save_folder = save_folder
        self.transform = transform
        self.target_transform = target_transform

        if recursive:
            # Find all raw_adc_*.npy files recursively
            self.x_paths = sorted(glob(os.path.join(save_folder, '**', 'raw_adc_*.npy'), recursive=True))
            self.y_paths = [p.replace('raw_adc_', 'range_doppler_map_') for p in self.x_paths]
        else:
            # Use indices to construct paths just like in the original version
            if indices is None:
                raise ValueError("indices must be provided when recursive=False")
            self.x_paths = [os.path.join(save_folder, f'raw_adc_{i}.npy') for i in indices]
            self.y_paths = [os.path.join(save_folder, f'range_doppler_map_{i}.npy') for i in indices]

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = np.load(self.x_paths[idx])  # shape: (512, 256, 16), dtype: complex64
        y = np.load(self.y_paths[idx])  # shape: ?, dtype: complex64 or float

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


class RadarFFTDataset(Dataset):
    def __init__(self, save_folder, indices, transform=None, target_transform=None):
        self.save_folder = save_folder
        self.adc_folder=save_folder+'/ADC'
        #self.fft_folder=save_folder+'/FFT'
        self.fft_range_dopller=save_folder+'/FFT2'
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        
        x = np.load(os.path.join(self.adc_folder, f'raw_adc_{i}.npy'))   
        y = np.load(os.path.join(self.fft_range_dopller, f'second_fft_{i}.npy'))  

        
        x = torch.from_numpy(x)  
        y = torch.from_numpy(y)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y




def load_data(save_folder, indices):
    dataset = RadarDataset(save_folder, indices)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
    return dataloader

def split_dataloader(dataset, batch_size=8, num_workers=4, pin_memory=True,
                     train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                     shuffle=True, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate shuffled indices
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    # Compute split sizes
    n_total = len(indices)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Split
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader



#only for testing
def main():
    sequence = 'RECORD@2020-11-21_11.54.31'
    save_folder = f'/media/christophe/backup/DATARADIAL/{sequence}'

    
    indices = list(range(250))

    dataset = RadarDataset(save_folder, indices)
    print(f"Dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
    print(f"DataLoader length: {len(dataloader)}")
    count = 0
    for x, y in dataloader:
        print(x.shape, x.dtype)  # e.g., torch.Size([8, 512, 256, 16]) torch.complex64
        print(y.shape,y.dtype)
        count += 1
        if count > 1:
            break
    train_loader1, val_loader1, test_loader1 = split_dataloader(dataset, seed=123)
    train_loader2, val_loader2, test_loader2 = split_dataloader(dataset, seed=123)

    assert list(train_loader1.dataset.indices) == list(train_loader2.dataset.indices)
    assert list(val_loader1.dataset.indices) == list(val_loader2.dataset.indices)
    assert list(test_loader1.dataset.indices) == list(test_loader2.dataset.indices)
    print("Train, val, and test loaders are identical across runs with the same seed.")

if __name__ == "__main__":
    main()