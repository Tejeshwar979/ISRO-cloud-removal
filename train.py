# optimizer , training , loss fnx  

import torch 
import torch.nn as nn 
import torch.optim as optim 
from utils.model import *  
from utils.loader import * 
from utils.CARLloss import *  
from utils.loader import *  
from torch.utils.data import DataLoader 

# parser   

device = torch.device("cuda")  

import argparse   

parser = argparse.ArgumentParser()   

parser.add_argument("--epochs" , type = int , default = 1) 
parser.add_argument("--exp" , type = int , default = 1) 
parser.add_argument("--sar" , type = str , default = r"D:\ROIs1868_summer_s1\ROIs1868_summer_s1\s1_27") 
parser.add_argument("--cloud" , type = str , default = r"D:\ROIs1868_summer_s2_cloudy\ROIs1868_summer_s2_cloudy\s2_cloudy_27")
parser.add_argument("--target" , type = str , default = r"D:\ROIs1868_summer_s2\ROIs1868_summer_s2\s2_27")

args = parser.parse_args()   

# Dataset and Loader    

data = Data_preprocessing(sar = args.sar , cloud = args.cloud , target = args.target)  

Load_data = DataLoader(data , shuffle = True , batch_size = 4 )

model = Model().to(device)

c_optim = optim.Adam(model.parameters() , lr = 7e-5)    

CARLloss = CARLLoss().to(device)    

for epoch in range(args.epochs) : 
    for sar , cloud , target , cloudmask in (Load_data) :
        
        sar = sar.to(device)
        cloud = cloud.to(device)
        target = target.to(device)
        cloudmask = cloudmask.to(device)

        c_optim.zero_grad()
    
        pred = model(sar , cloud)
        pix_loss = CARLloss(target , pred , cloud , cloudmask )

        pix_loss.backward()

        c_optim.step()

    print(f" {epoch + 1} / {args.epochs} loss = {pix_loss} ")

torch.save(model.state_dict() , f"saved_model_{args.exp}.pth")


