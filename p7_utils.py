import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#from neuralop.layers.spectral_convolution import SpectralConv2d
import numpy as np


def create_dataloaders(real_data, ra_maps, batch_size=32, test_split=0.2, val_split=0.1, random_seed=42):
    """
    Crée des DataLoaders pour l'entraînement, la validation et le test
    

    """
    # Pour la reproductibilité
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Créer des indices et les mélanger
    dataset_size = len(real_data)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # Calculer les points de séparation
    test_size = int(np.floor(test_split * dataset_size))
    val_size = int(np.floor(val_split * dataset_size))
    train_size = dataset_size - test_size - val_size
    
    # Diviser les indices
    train_indices = indices[test_size + val_size:]
    val_indices = indices[test_size:test_size + val_size]
    test_indices = indices[:test_size]
    
    # Création des sous-ensembles
    train_real_data = [real_data[i] for i in train_indices]
    train_ra_maps = [ra_maps[i] for i in train_indices]
    
    val_real_data = [real_data[i] for i in val_indices]
    val_ra_maps = [ra_maps[i] for i in val_indices]
    
    test_real_data = [real_data[i] for i in test_indices]
    test_ra_maps = [ra_maps[i] for i in test_indices]
    
    
    train_dataset = RadarDataset(train_real_data, train_ra_maps)
    val_dataset = RadarDataset(val_real_data, val_ra_maps)
    test_dataset = RadarDataset(test_real_data, test_ra_maps)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Taille des ensembles - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader



class RadarDataset(Dataset):
    """
    Dataset personnalisé pour les données radar
    Input: real_data (forme: [512, 256, 16])
    Label: ra_maps (forme: [512, 751])
    """
    def __init__(self, real_data, ra_maps, transform=None):
        """
        Initialisation du dataset
        
        Args:
            real_data: Liste de tableaux numpy de forme (512, 256, 16)
            ra_maps: Liste de tableaux numpy de forme (512, 751)
            transform: Transformations optionnelles à appliquer aux données
        """
        self.real_data = real_data
        self.ra_maps = ra_maps
        self.transform = transform
        
        # Vérification que les listes ont la même longueur
        assert len(self.real_data) == len(self.ra_maps), "Les listes d'entrées et d'étiquettes doivent avoir la même longueur"
    
    def __len__(self):
        """Retourne la taille du dataset"""
        return len(self.real_data)
    
    def __getitem__(self, idx):
        """
        Retourne une paire (entrée, étiquette) à l'indice idx
        """
        # Récupère les données à l'indice idx
        input_data = self.real_data[idx]
        label = self.ra_maps[idx]
        
        
        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(label)
        
        # Applique les transformations si elles existent
        if self.transform:
            input_tensor = self.transform(input_tensor)
            
        return input_tensor, label_tensor

# CNN able to produce range andgle maps
class FixedSpectralNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.preprocess = nn.Conv2d(16, 32, kernel_size=1)
        
        self.spectral_conv1 = SpectralConv2d(
            in_channels=32,
            out_channels=64,
            n_modes=(32, 32) 
        )
        
        self.spectral_conv2 = SpectralConv2d(
            in_channels=64,
            out_channels=128,
            n_modes=(16, 16)
        )
        
        self.output_projection = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  
        )
        
        self.final_reshape = nn.Linear(256, 751)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
    
        x = x.permute(0, 3, 1, 2)
        
        x = self.preprocess(x) 
        
        x = self.spectral_conv1(x)  
        x = torch.relu(x)
        
        x = self.spectral_conv2(x) 
        x = torch.relu(x)
        
        
        x = self.output_projection(x)  
    
        x = x.squeeze(1)  

        x = self.final_reshape(x)  
        
        return x

def plot_network_loss(losses,file_name):
    """
    Plot the training loss over iterations.
    
    Args:
        losses (list): List of loss values recorded during training.
    """
    
    if not losses:
        print("No loss values to plot.")
        return
    
    plt.figure(figsize=(10, 6))

    plt.plot(losses, color='blue', linewidth=2)

    plt.title('Évolution de la perte pendant l\'entraînement', fontsize=16)
    plt.xlabel('Itérations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig(file_name, dpi=300)

def list_record_folders(path_d1):
    """
        
    Returns:
        list: List of folders whose name starts with RECORD@
    """
    result = []
    
    
    if not os.path.isdir(path_d1):
        print(f"Error: {path_d1} is not a valid directory")
        return result
        
    
    for item in os.listdir(path_d1):
        full_path = os.path.join(path_d1, item)
        
        
        if os.path.isdir(full_path) and item.startswith("RECORD@"):
            result.append(path_d1+item)
            
    return result

def __main__():
    # Example usage
    path_d1 = '/home/christophe/RADIalP7/DATASET/'
    folders = list_record_folders(path_d1)
    print(folders)
