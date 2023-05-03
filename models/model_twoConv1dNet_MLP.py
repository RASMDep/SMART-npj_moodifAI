import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor
from loss_functions.losses import AsymmetricLoss,ASLSingleLabel
import torch.nn.functional as F
#from balanced_loss import Loss
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# Define the GRU
class GRU(nn.Module):
    def __init__(self, input_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=32, num_layers=2, batch_first=True)
    
    def forward(self, x):
        _, x = self.gru(x)
        return x[-1]


class twoConv1dNet_MLP(torch.nn.Module):

    def __init__(self,args):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_size1 = 4 #args.n_channels
        self.in_size2 = 3 #args.n_channels
        self.n_out_class = args.num_class
        self.n_feat = args.n_covariates
        self.less_features = args.less_features

        if args.loss == 'ASL':
            self.criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # ECG and ACC
        self.feature_extractor1 = nn.Sequential(
                        nn.Conv1d(self.in_size1, 16, 5, stride=1, padding = 2),nn.ReLU(),nn.MaxPool1d(4, stride=4),nn.BatchNorm1d(16),
                        nn.Conv1d(16, 32, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(32),
                        nn.Conv1d(32, 64, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(64),
                        nn.Conv1d(64, 128, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(128),
                        nn.Conv1d(128, 256, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4))#, nn.BatchNorm1d(256),
                        ## downsampling of max to this size and write the pooling 
                        #nn.Conv1d(256, 512, 5, stride=1, padding = 2), nn.ReLU())
    

        self.reduce_features1 = nn.Sequential(
                        #nn.Conv1d(512, 256, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(256, 128, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(128, 64, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(64, 32, 5, stride=1, padding = 2),nn.ReLU()
                        )
        
        self.lastpool1 = nn.AdaptiveMaxPool1d(1) 

        self.gru1 = GRU(256)

        ## GPS

        self.feature_extractor2 = nn.Sequential(
                        nn.Conv1d(self.in_size2, 8, 3, stride=1, padding = 2),nn.ReLU(),nn.MaxPool1d(2, stride=2),nn.BatchNorm1d(8),
                        nn.Conv1d(8, 16, 3, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(16),
                        nn.Conv1d(16, 32, 3, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(32),
                        nn.Conv1d(32, 64, 3, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(64),
                        nn.Conv1d(64, 128, 3, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(2, stride=2))#, nn.BatchNorm1d(128),
                        ## downsampling of max to this size and write the pooling 
                        #nn.Conv1d(256, 512, 5, stride=1, padding = 2), nn.ReLU())
    

        self.reduce_features2 = nn.Sequential(
                        #nn.Conv1d(512, 256, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(256, 128, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(128, 64, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(64, 32, 5, stride=1, padding = 2),nn.ReLU()
                        )
        
        self.lastpool2 = nn.AdaptiveMaxPool1d(1) 

        self.gru2 = GRU(128)


        self.classifier_task1 = nn.Sequential(
                    nn.Linear(32*2, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class ))
       
        self.classifier_task2 = nn.Sequential(
                    nn.Linear(32*2, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class))
        
        self.classifier_task3 = nn.Sequential(
                    nn.Linear(32*2, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class))
    

    def forward(self, sample):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        
        """
        
        x1_t = sample["x1_t"]
        x1_t= x1_t.to(self.device)
        x_ext1_t = torch.zeros((x1_t.shape[0], x1_t.shape[1], int(np.ceil(x1_t.shape[2]/256))*256)).to(self.device)
        x_ext1_t[:,:, 0:x1_t.shape[2]] = x1_t

        ### divide in 5 minutes, needed only when using the GRU
        batch_size, num_channels, signal_length = x_ext1_t.size()# GRU 
        x_ext1_t = x_ext1_t.view(batch_size * (signal_length // 38400), num_channels, 38400)# GRU 

        x1_ = sample["x1_"]
        x1_= x1_.to(self.device)
        x_ext1_ = torch.zeros((x1_.shape[0], x1_.shape[1], int(np.ceil(x1_.shape[2]/256))*256)).to(self.device)
        x_ext1_[:,:, 0:x1_.shape[2]] = x1_

        ### divide in 5 minutes, needed only when using the GRU
        batch_size, num_channels, signal_length = x_ext1_.size()# GRU 
        x_ext1_ = x_ext1_.view(batch_size * (signal_length // 38400), num_channels, 38400)# GRU 

        x2_t = torch.nan_to_num(sample["x2_t"])
        x2_t= x2_t.to(self.device)
        x_ext2_t = torch.zeros((x2_t.shape[0], x2_t.shape[1],x2_t.shape[2])).to(self.device)
        x_ext2_t[:,:, 0:x2_t.shape[2]] = x2_t

        x2_ = torch.nan_to_num(sample["x2_"])
        x2_= x2_.to(self.device)
        x_ext2_ = torch.zeros((x2_.shape[0], x2_.shape[1],x2_.shape[2])).to(self.device)
        x_ext2_[:,:, 0:x2_.shape[2]] = x2_


        z1_t = self.feature_extractor1(x_ext1_t)
        z1_t = z1_t.view(batch_size, -1, 256, ) # GRU

        z1_ = self.feature_extractor1(x_ext1_)
        z1_ = z1_.view(batch_size, -1, 256, ) # GRU

        z1 = z1_t - z1_
        
        z1 = self.gru1(z1)                   # GRU              
        
        z2_t = self.feature_extractor2(x_ext2_t)
        z2_t = z2_t.view(batch_size, -1, 128, ) # GRU
        
        z2_ = self.feature_extractor2(x_ext2_)
        z2_ = z2_.view(batch_size, -1, 128, ) # GRU

        z2 = z2_t - z2_
        
        z2 = self.gru2(z2)                   # GRU 

        z = torch.cat((z1,z2),1)             
        
        arousal = self.classifier_task1(z)
        valence = self.classifier_task2(z)
        time = self.classifier_task3(z)
     

        return arousal, valence, time


    def loss(self, y_pred, y_gt):
        return self.criterion(y_pred, y_gt )
         

