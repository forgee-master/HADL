import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.mylowrank import LowRank
from layers.dct import DiscreteCosineTransform as DCT

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.rank = configs.rank

        self.channels = configs.enc_in

        self.low_pass_filter = torch.tensor([1, 1], dtype=torch.float32) / math.sqrt(2)

        self.low_pass_filter = self.low_pass_filter.reshape(1,1,-1).repeat(self.channels, 1, 1)


        if (self.seq_len%2)!=0:
            in_len = self.seq_len//2 + 1
        else:
            in_len = self.seq_len//2


        ## Full Linear Layer
        # self.layer_lo = nn.Linear(in_len,self.pred_len)

        ## Feed-Forward Network
        # self.layer_lo = FFN(in_len, 32, self.pred_len)
        
        ## Low Rank Self-Gradient Calculation with Adam Optimizier.
        ## This method does not follow strict orthogonality.
        # self.layer_lo = ThinLinear(in_features=in_len,
        #                            out_features=self.pred_len,
        #                            rank=self.rank,
        #                            bias=False)


        ## Jonas's Low Rank
        ## Ensure to switch-off Adam Optimizer in "exp_main" train function.
        self.layer_lo = nn.Linear(in_features=in_len,
                                out_features=self.pred_len,
                                bias=True)



    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch, _, _ = x.shape

        ## Scaled Normalization
        x = x.permute(0,2,1)
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean
        
        ## Haar decomposition
        if self.seq_len%2 != 0:
            x = F.pad(x, (0,1))

        x = F.conv1d(input=x, weight=self.low_pass_filter, stride=2, groups=self.channels)

        ##Cosine Transform
        x = DCT.apply(x) / x.shape[-1]

        ## Prediction
        out = self.layer_lo(x)

        out = out + seq_mean

        return out.permute(0,2,1) # [Batch, Output length, Channel]
