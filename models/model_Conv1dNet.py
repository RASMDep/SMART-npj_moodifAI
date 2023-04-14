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


class Conv1dNet(torch.nn.Module):

    def __init__(self,args):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_size = args.n_channels
        self.n_out_class = args.num_class
        self.n_feat = args.n_covariates
        self.less_features = args.less_features


        if args.loss == 'ASL':
            self.criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 12 is the input space in terms of ECG leads, 16 is the output space corresponding to the new features
        # Conv1d(input size==ecg channels, outputs size , filter size)
        self.feature_extractor = nn.Sequential(
                        nn.Conv1d(self.in_size, 16, 5, stride=1, padding = 2),nn.ReLU(),nn.MaxPool1d(4, stride=4),nn.BatchNorm1d(16),
                        nn.Conv1d(16, 32, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(32),
                        nn.Conv1d(32, 64, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(64),
                        nn.Conv1d(64, 128, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.BatchNorm1d(128),
                        nn.Conv1d(128, 256, 5, stride=1, padding = 2), nn.ReLU(), nn.MaxPool1d(4, stride=4))#, nn.BatchNorm1d(256),
                        ## downsampling of max to this size and write the pooling 
                        #nn.Conv1d(256, 512, 5, stride=1, padding = 2), nn.ReLU())
    

        self.reduce_features = nn.Sequential(
                        #nn.Conv1d(512, 256, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(256, 128, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(128, 64, 5, stride=1, padding = 2),nn.ReLU(),
                        nn.Conv1d(64, 32, 5, stride=1, padding = 2),nn.ReLU()
                        )
        
        self.lastpool = nn.AdaptiveMaxPool1d(1) 

        self.gru = GRU(256)


        self.classifier_task1 = nn.Sequential(
                    nn.Linear(32+self.n_feat, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class ))
       
        self.classifier_task2 = nn.Sequential(
                    nn.Linear(32+self.n_feat, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class))
        
        self.classifier_task3 = nn.Sequential(
                    nn.Linear(32+self.n_feat, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class))
        
        self.classifier_task4 = nn.Sequential(
                    nn.Linear(32+self.n_feat, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class))
    
        self.classifier_task5 = nn.Sequential(
                    nn.Linear(32+self.n_feat, 16),nn.ReLU(),
                    nn.Linear(16, self.n_out_class))
    

        # used when transformining into a n classifier x features version
        # arousal classifier
        self.block8_arosual1 = nn.Sequential(nn.Linear(256+self.n_feat, 512),nn.ReLU())
        self.p7_arosual1 = nn.Parameter(torch.randn((512,32,self.n_out_class)),requires_grad=True)
        self.p8_arosual1 = nn.Parameter(torch.randn((32, self.n_out_class)), requires_grad=True) 
        # valence classifier
        self.block8_valence1 = nn.Sequential(nn.Linear(256+self.n_feat, 512),nn.ReLU())
        self.p7_valence1 = nn.Parameter(torch.randn((512,32,self.n_out_class)),requires_grad=True)
        self.p8_valence1 = nn.Parameter(torch.randn((32, self.n_out_class)), requires_grad=True) 
        # arousal classifier
        self.block8_arosual2 = nn.Sequential(nn.Linear(256+self.n_feat, 512),nn.ReLU())
        self.p7_arosual2 = nn.Parameter(torch.randn((512,32,self.n_out_class)),requires_grad=True)
        self.p8_arosual2 = nn.Parameter(torch.randn((32, self.n_out_class)), requires_grad=True) 
        # valence classifier
        self.block8_valence2 = nn.Sequential(nn.Linear(256+self.n_feat, 512),nn.ReLU())
        self.p7_valence2 = nn.Parameter(torch.randn((512,32,self.n_out_class)),requires_grad=True)
        self.p8_valence2 = nn.Parameter(torch.randn((32, self.n_out_class)), requires_grad=True) 
        
        

    def forward(self, sample):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        
        """
        
        x = sample["x"]
        x= x.to(self.device)
        x_ext = torch.zeros((x.shape[0], x.shape[1], int(np.ceil(x.shape[2]/256))*256)).to(self.device)
        x_ext[:,:, 0:x.shape[2]] = x

        ### divide in 5 minutes, needed only when using the GRU
        batch_size, num_channels, signal_length = x_ext.size()# GRU 
        x_ext = x_ext.view(batch_size * (signal_length // 38400), num_channels, 38400)# GRU 
        
        
        if self.less_features == True:
        
            z = self.feature_extractor(x_ext)
            #z = self.reduce_features(z)      # simple CNN
            z = z.view(batch_size, -1, 256, ) # GRU
            z = self.gru(z)                   # GRU              
            #z = self.lastpool(z).squeeze(2)  # simple CNN
            arosual1 = self.classifier_task1(z)
            valence1 = self.classifier_task2(z)
            time = self.classifier_task3(z)
            arosual2 = self.classifier_task4(z)
            valence2 = self.classifier_task5(z)
           
        else:
            z = self.feature_extractor(x_ext)
            z = self.lastpool(z).squeeze(2)
        
            arosual = self.block8_arosual(z)
            arosual = torch.einsum("bf, fec->bec",arosual,self.p7_arosual)
            arosual= F.relu(arosual)
            arosual = torch.einsum("bec,ec->bc",arosual,self.p8_arosual)          

            valence = self.block8_valence(z)
            valence = torch.einsum("bf, fec->bec",valence,self.p7_valence)
            valence = F.relu(valence)
            valence = torch.einsum("bec,ec->bc",valence,self.p8_valence)      

        return arosual1,valence1, time, arosual2, valence2


    def loss(self, y_pred, y_gt):
        return self.criterion(y_pred, y_gt )
         

