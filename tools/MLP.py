
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, num_in, num_out):
        super().__init__()
        
        self.network = nn.Sequential(nn.Linear(num_in, 32),
                                     nn.ReLU(),
                                    #  nn.Dropout(0.2),
                                     nn.Linear(32, 64),
                                     nn.ReLU(),
                                    #  nn.Dropout(0.2),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                    #  nn.Dropout(0.2),
                                     nn.Linear(64, num_out),
                                     nn.Sigmoid())


    def forward(self, x):
        return self.network(x)