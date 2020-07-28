from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        

    def forward(self, input):
        out = F.normalize(input)
        return out