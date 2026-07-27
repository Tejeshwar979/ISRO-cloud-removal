# create a loader for images 

from PIL import Image 

import torch 
import torch.nn as nn  

from torchvision import transforms 

import os 



class Data_preprocessing : 
    def __init__(self , root_dir_path ):
        self.root_dir_path = root_dir_path  
        pass  
