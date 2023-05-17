
import torch
import torch.nn as nn


# def RMSELoss(predict, target):
#     return torch.sqrt(torch.mean((predict-target)**2))

criteria = nn.MSELoss()

def RMSELoss(predict, target):
    return torch.sqrt(criteria(predict, target))