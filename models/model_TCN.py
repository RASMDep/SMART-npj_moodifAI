# Script defines the TCN model to train the network on

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from loss_functions.losses import AsymmetricLoss

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()

        #self.criterion = nn.BCEWithLogitsLoss()
        if args.loss == 'ASL':
            self.criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sigm = False

        self.input_size = args.n_channels
        self.output_size = args.num_class
        self.num_channels = [3, 5, 5, 5, 6, 6]
        self.kernel_size = 5
        self.dropout = 0.2

        self.tcn = TemporalConvNet(self.input_size, self.num_channels, kernel_size=self.kernel_size, dropout=self.dropout)
        self.global_residual = nn.Conv1d(self.input_size, self.output_size, kernel_size=1)  # Global skip from input to last layer
        self.out = nn.Linear(self.num_channels[-1]+self.input_size, self.output_size)
        self.weight_init_res()

    def weight_init_res(self):
        self.global_residual.weight.data.normal_(0, 0.01)

    def forward(self, sample):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = sample["x"]
        x= x.to(self.device)
        y = self.tcn(x)  # input should have dimension (N, C, L)
        y = torch.cat((y, x), dim=1)
        out = self.out(y.transpose(1, 2)).transpose(1, 2)
        out = out[:,:,-1]
        #out = self.out(y)

        return out,0,0

    
    def loss(self, y_pred, y_gt):
        
        return self.criterion(y_pred, y_gt)