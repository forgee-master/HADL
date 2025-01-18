import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                    self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach().to(x.device)
        x = x - seq_last

        x = x.permute(0,2,1)

        if self.individual:
             output = torch.zeros([x.size(0),x.size(1),self.pred_len],dtype=x.dtype).to(x.device)
             for i in range(self.channels):
                output[:,i,:] = self.Linear[i](x[:,i,:])
            
        else:
            x = self.Linear(x)

        x = x.permute(0,2,1) + seq_last
        return x # [Batch, Output length, Channel]