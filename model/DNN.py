import torch
import torch.nn.functional as F
import torch.nn as nn

# DNN 
class CNN(nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2), #channel of ZPI is 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, dim_out, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.MaxPool2d(5,5)

    def forward(self, zigzag_window_PI):
        feature = self.features(zigzag_window_PI)
        feature = self.maxpool(feature)
        feature = feature.view(-1, self.dim_out) #B, dim_out
        return feature