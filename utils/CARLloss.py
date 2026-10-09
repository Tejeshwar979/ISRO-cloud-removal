import torch 
import torch.nn as nn  

device = torch.device("cuda")

class CARLLoss(nn.Module) : 
    def __init__(self , lsm = 1):

        super().__init__()

        self.lsm = lsm 

    def forward(self , target , pred , input , csm) : 

        target = target.to(device) 
        pred = pred.to(device) 
        input = input.to(device)
        csm = csm.to(device)

        csm1 = csm * (pred - target) 

        csm2 = (1 - csm) * (pred - input) 

        L1Loss = torch.abs(pred - target)  

        loss = (torch.abs(csm1 + csm2) + self.lsm * (L1Loss)).mean()

        return loss 


