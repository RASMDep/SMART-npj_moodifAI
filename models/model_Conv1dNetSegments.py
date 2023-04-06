import numpy as np
import torch
import torch.nn as nn


class Conv1dNetSegments(torch.nn.Module):

    def __init__(self,args):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()

        # 12 is the input space in terms of ECG leads, 16 is the output space corresponding to the new features
        # Conv1d(input size==ecg channels, outputs size , filter size)
        self.block1 = nn.Sequential(nn.Conv1d(12, 16, 5, stride=1, padding = 2),nn.ReLU(),nn.MaxPool1d(4, stride=4),nn.BatchNorm1d(16))
        self.block2 = nn.Sequential(nn.Conv1d(16, 32, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(32))
        self.block3 = nn.Sequential(nn.Conv1d(32, 64, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(64))
        self.block4 = nn.Sequential(nn.Conv1d(64, 128, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(128))
        self.block5 = nn.Sequential(nn.Conv1d(128, 256, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(256))
        self.block6 = nn.Sequential(nn.Conv1d(256, 27, 1), nn.Sigmoid()) 


    def forward(self, x, covariates):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = self.block5(z)
        z = self.block6(z)

        z = torch.log(1e-7+1-z)
        z = torch.sum(z, dim=2)
        z = torch.exp(z)
        z = torch.log(torch.clamp(1-z, 1e-7, 1-1e-7))  # probability that one of the class is present in at least one of the 10s windows, for each class
        
        assert(not torch.isnan(torch.sum(z)))

        return z 

    def loss(self, y_pred, y_gt):
        
        return self.criterion(y_pred, y_gt)
         