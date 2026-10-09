# create a loader for images   

import rasterio
import torch 
import torch.nn as nn  
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_opening, binary_closing
from skimage.morphology import disk
from torchvision import transforms 

import os 

device = torch.device("cuda")

def create_mask(cloud, target):

    diff = torch.abs(target - cloud)

    diff = diff.mean(dim=0)

    diff = diff.detach().cpu().numpy()

    threshold = threshold_otsu(diff)

    mask = diff > threshold

    kernel = disk(3)

    mask = binary_opening(mask, structure=kernel)
    mask = binary_closing(mask, structure=kernel)

    mask = torch.from_numpy(mask.astype(np.float32))

    mask = mask.unsqueeze(0)

    return mask



class Data_preprocessing : 
    def __init__(self , sar , cloud , target ):
        self.sar = sar
        self.cloud = cloud 
        self.target = target  
        self.transformations = transforms.Compose([
            transforms.CenterCrop(512) , 
            transforms.ToTensor() , 
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
        ])  
        self.sar_dataset = [os.path.join(sar , img) for img in os.listdir(self.sar)]
        self.cloud_dataset = [os.path.join(cloud , img) for img in os.listdir(self.cloud)]
        self.target_dataset = [os.path.join(target , img) for img in os.listdir(self.target)]
    
    def __len__(self):
        sar_len = len(self.sar_dataset)
        cloud_len = len(self.cloud_dataset)
        target_len = len(self.target_dataset)
        return min(
            sar_len , 
            cloud_len , 
            target_len
        )

    def __getitem__(self, idx):

        sar_path = self.sar_dataset[idx]
        cloud_path = self.cloud_dataset[idx]
        target_path = self.target_dataset[idx]


        with rasterio.open(sar_path) as src:
            sar_img = src.read()


        with rasterio.open(cloud_path) as src:
            cloud_img = src.read()


        with rasterio.open(target_path) as src:
            target_img = src.read()


        sar_img = torch.from_numpy(
            sar_img.astype(np.float32)
        )

        cloud_img = torch.from_numpy(
            cloud_img.astype(np.float32)
        )

        target_img = torch.from_numpy(
            target_img.astype(np.float32)
        )

        mask_img = create_mask(
            cloud_img,
            target_img
        )

        return sar_img, cloud_img, target_img, mask_img


