# create resnet architecture  
import torch 
import torch.nn as nn  
from PIL import Image  

# init_conv layer 

class f0(nn.Module) : 
    def __init__(self):
        self.init_features = nn.Sequential(
            nn.Conv2d(15 , 64 , 3 , 1 , 1),
            nn.ReLU(),
        )
    def forward(self , x1 , x2):
        x = torch.cat([x1 , x2] , dim = 1)
        return self.init_features(x)

# Res Block

class Resblock(nn.Module) : 
    def __init__(self, res_scale = 0.1):
        super().__init__()
        self.Reslayer = nn.Sequential(
            nn.Conv2d(64 , 128 , 3 , 1, 1),
            nn.ReLU(),
            nn.Conv2d(128 , 256 , 3 , 1 , 1) ,
            nn.ReLU(),
        )
        self.scale = res_scale
    def forward(self , x):
        out = self.Reslayer(x)
        scaled_out = out * self.scale   
        return x + scaled_out 


class ResLayers(nn.Module) : 
    def __init__(self):
        super().__init__()
        self.layers_Res = nn.Sequential(
            *[Resblock() for _ in range(10)] 
        )
    def forward(self , x) : 
        return self.layers_Res(x)


class Model(nn.Module) : 
    def __init__(self):
        super().__init__()
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 , 256 , 3 , 1, 1)
        )
    def forward(self , x1 , x2):
        out = f0(x1 , x2)
        out = ResLayers(out)
        return x2 +  self.final_conv(out)


