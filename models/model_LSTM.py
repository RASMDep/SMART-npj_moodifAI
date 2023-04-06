import numpy as np
import torch
import torch.nn as nn


class LSTM(torch.nn.Module): # Thank you Python Engineer! https://www.youtube.com/watch?v=0_PgWWmauHk

    def __init__(self,args):
        super(LSTM, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.sigm = False

        self.num_layers = 2 # how many RNN layers stacked together
        self.hidden_channels = 128
        self.lstm = torch.nn.LSTM(12, self.hidden_channels, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_channels, 27)
    
    def forward(self, x):
        # x -> batch_size, seq_length, input_size 
        x = x.transpose(1,2)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)

        # Push through rnn (out: batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Take only hidden vector of last sequence
        out = out[:, -1, :]
        # Decode into classes
        out = self.linear(out)
        if self.sigm == True:
            out = torch.nn.functional.softmax(out, dim=-1) # Last layer same as ncde
        return out

    def loss(self, y_pred, y_gt):
        
        return self.criterion(y_pred, y_gt)