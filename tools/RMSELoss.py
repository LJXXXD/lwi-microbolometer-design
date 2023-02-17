
import torch


def RMSELoss(predict, target):
    return torch.sqrt(torch.mean((predict-target)**2))