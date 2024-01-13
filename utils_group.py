import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from PIL import Image

c45 = np.cos(np.pi / 4)
s45 = np.sin(np.pi / 4)
ROT1 = torch.tensor([[c45, -s45, 0], [s45, c45, 0], [0, 0, 1]], dtype=torch.float32)
a2 = np.arctan(np.sqrt(2))
ca2 = np.cos(a2)
sa2 = np.sin(a2)
ROT2 = torch.tensor([[1, 0, 0], [0, ca2, -sa2], [0, sa2, ca2]], dtype=torch.float32)
ROT21 = ROT2 @ ROT1
ROT21_INV = ROT21.inverse()

# utils to manage images
def rotate_hue(im, angle):
    '''
    Rotate the hue of an image by a given angle.
    Args:
        im (torch.Tensor): Image tensor of shape (batch_size, 3, height, width)
        angle (float): Angle to rotate hue by. Should be in range [0, 1)
    Returns:
        torch.Tensor: Image tensor of shape (batch_size, 3, height, width)'''
    if im.shape[-3] != 3:
        raise ValueError("Input must be 3-channel (HSV)")
    original_shape = im.shape
    im = im.reshape(-1, 3, *im.shape[-2:])
    im[:, 0] = (im[:, 0] + angle) % 1.0
    return im.reshape(original_shape)

def rotate_hue_matrix(im, angle):
    cangle = np.cos(angle)
    sangle = np.sin(angle)
    rot_hue = torch.tensor([[cangle, -sangle, 0], [sangle, cangle, 0], [0, 0, 1]], dtype=torch.float32)
    combined_rotation = ROT21_INV @ rot_hue @ ROT21
    im_shape = im.shape
    im = im.reshape(3, -1)
    im = combined_rotation @ im
    im = im.clip(0, 1)
    return im.reshape(im_shape)


def rotate_hue(im, angle, rgb_out, rgb_in=True):
    image_hsv = im.convert("HSV") if rgb_in else im
    h, s, v = image_hsv.split()
    h = h.point(lambda p: (p + angle) % 256)
    image_hsv = Image.merge("HSV", (h, s, v))
    # image_rgb = image_hsv.convert("RGB")
    if not rgb_out:
        return image_hsv
    else:
        return image_hsv.convert("RGB")

def scale_luminance(im, amt, rgb_out, rgb_in):
    image_hsv = im.convert("HSV") if rgb_in else im
    h, s, v = image_hsv.split()
    v = v.point(lambda p: min(255, max(0, (p + amt))))
    image_hsv = Image.merge("HSV", (h, s, v))

    if not rgb_out:
        return image_hsv
    else:
        return image_hsv.convert("RGB")

def rotate_value(im, power):
    image_hsv = im.convert("HSV")
    h, s, v = image_hsv.split()
    v = v.point(lambda p: ((p / 255.0) ** power) * 255)
    image_hsv = Image.merge("HSV", (h, s, v))
    image_rgb = image_hsv.convert("RGB")
    return image_rgb

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
