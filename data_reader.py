import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import  Subset


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
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
    return dataloader



def split_dataloader(dataset, batch_size=2, num_workers=4, pin_memory=True,
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
        if count > 10:
            break
    

if __name__ == "__main__":
    main()