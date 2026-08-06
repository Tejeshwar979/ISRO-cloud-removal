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

import argparse

parser = argparse.ArgumentParser() 

parser.add_argument("--epochs" , type = int , default = 10) 
parser.add_argument("--exp" , type = int , default = 1) 

parser.add_argument("--sar" , type = str , default = "") 
parser.add_argument("--cloud" , type = str , default = "")
parser.add_argument("--target" , type = str , default = "")

args = parser.parse_args()

# Dataset and Loader  

data = DataLoader(sar = args.sar , cloud = args.cloud , target = args.target , shuffle = True , batch_size = 16 )

model = Model()

c_optim = optim.Adam(model.parameters() , lr = 7e-5) 

DataLoader = None

CARLloss = CARLLoss()

for epoch in range(args.epochs) : 
    for sar , cloud , target , cloudmask in data :

        c_optim.zero_grad()
        pred = model(sar , cloud)

        pix_loss = CARLloss(target , pred , cloud , cloudmask )
        pix_loss.backward()
        c_optim.step()

    print(f" {epoch + 1}/ {args.epochs} loss = {pix_loss} ")

torch.save(model.state_dict() , f"saved_model_{args.exp}.pth")


