# create a loader for images 

from PIL import Image 

import torch 
import torch.nn as nn  
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_opening, binary_closing
from skimage.morphology import disk
from torchvision import transforms 

import os 

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
    def __init__(self , root_dir_path ):
        self.root_dir_path = root_dir_path  
        pass  


