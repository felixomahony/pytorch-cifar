import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from PIL import Image
import parquet

class SmallNorbDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.length = len(data)
    
    def __getitem__(self, index):
        x = self.data[index]

        
        if self.transform:
            x = self.transform(x)
        
        y = self.targets[index]
        
        return x, y

    def __len__(self):
        return self.length

class DSpritesDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.length = data.shape[0]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index, 1]
        
        if self.transform:
            x = Image.fromarray(x * 255)
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return self.length

# dataset classes
class HDF5Dataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)  # Number of samples in the HDF5 file

    def __getitem__(self, idx):
        sample_data = self.imgs[idx]  # Access the dataset at the specified index
        data = Image.fromarray(sample_data)
        sample_label = self.labels[idx]  # Access the dataset at the specified index
        label = torch.tensor(sample_label, dtype=torch.long)  # Assuming 'label' is a dataset in your HDF5 file

        if self.transform:
            data = self.transform(data)
        l = label[4]
        return data, l