
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor
from loss_functions.losses import AsymmetricLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Conv1dNet(torch.nn.Module):

    def __init__(self,args):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.in_size = args.n_channels
        self.n_out_class = args.num_class
        self.n_feat = args.n_covariates

        #self.criterion = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor([0.7]))
        self.criterion = nn.CrossEntropyLoss()

        # Define the attention mechanism
        self.attention_layer = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        # 12 is the input space in terms of ECG leads, 16 is the output space corresponding to the new features
        # Conv1d(input size==ecg channels, outputs size , filter size)
        self.block1 = nn.Sequential(nn.Conv1d(self.in_size, 16, 5, stride=1, padding = 2),nn.ReLU(),nn.MaxPool1d(2, stride=2),nn.BatchNorm1d(16))
        self.block2 = nn.Sequential(nn.Conv1d(16, 32, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(32))
        self.block3 = nn.Sequential(nn.Conv1d(32, 64, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(64))
        self.block4 = nn.Sequential(nn.Conv1d(64, 128, 5, stride=1, padding = 2), nn.ReLU(), nn.BatchNorm1d(128))# nn.MaxPool1d(2, stride=2)
        ## downsampling of max to this size and write the pooling 
        self.block5 = nn.Sequential(nn.Conv1d(128, 256, 5, stride=1, padding = 2), nn.ReLU())

        self.lastpool = nn.AdaptiveMaxPool1d(1) 
        self.block6 = nn.Sequential(nn.Linear(256+self.n_feat, 512),nn.ReLU()) 
        # go from 64x4 to 1 with linear layer
        self.l6 = nn.Conv1d(512, self.n_out_class, 1, stride=1, padding = 0)
        self.l7 = nn.Linear(512, self.n_out_class )



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

        # Apply attention mechanism
        z = z.permute(2, 0, 1)  # Change the order of dimensions
        z, attention_weights = self.attention_layer(z, z, z)  # Self-attention
        z = z.permute(1, 2, 0)  # Change the order of dimensions back

        plotting=0
        if plotting==1:

            # Choose a specific attention head to visualize
            head_to_visualize = 0

            # Plot the attention map for a specific head
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.imshow(attention_weights[head_to_visualize].detach().cpu().numpy(), aspect='auto')
            plt.title(f'Attention Map - Head {head_to_visualize}')
            plt.xlabel('Input Sequence Position')
            plt.ylabel('Output Sequence Position')

            # Overlay attention weights on the original input
            plt.subplot(2, 1, 2)
            plt.plot(x[4].detach().cpu().numpy().transpose(), label='Original Input')
            plt.title('Original Input with Attention Weights')
            plt.xlabel('Input Sequence Position')
            plt.legend()

            plt.show()





        #z = z*mask_d + (1-mask_d)*(-1e6)
        z = self.lastpool(z).squeeze(2)
        if self.n_feat>0:
            # add additional hand crafted features before last layer, do not forget to normalize these features (0 mean, 1 variance)
            z = torch.cat([z,covariates], dim = 1)
        # Finally: multy layer percepton with one hidden layer and 1 linear layer for the classification
        z = self.block6(z)





        z = self.l7(z)

        return z 


    def loss(self, y_pred, y_gt):
        return self.criterion(y_pred, y_gt)

