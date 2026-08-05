# optimizer , training , loss fnx 

from utils.model import *  
from utils.loader import * 
from utils.CARLloss import *  

import argparse 

parser = argparse.ArgumentParser() 

parser.add_argument("--epochs" , type = int , default = 10) 

args = parser.parse_args()

model = Model()

import torch 
import torch.nn as nn 
import torch.optim as optim  

c_optim = optim.Adam(model.parameters() , lr = 7e-5) 

DataLoader = None

CARLloss = CARLLoss()


for epoch in range(args.epochs) : 
    for sar , cloud , target , cloudmask in DataLoader :

        c_optim.zero_grad()
        pred = model(sar , cloud)

        pix_loss = CARLloss(target , pred , cloud , cloudmask )
        pix_loss.backward()
        c_optim.step()

    print(f" {epoch +1 }/ {args.epochs} loss = {pix_loss} ")

torch.save(model.state_dict() , "saved_model_1.pth")


