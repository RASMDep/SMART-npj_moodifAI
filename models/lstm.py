import torch
import torch.nn as nn

class LSTMNet(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        self.input_size = 288
        self.hidden_size = 64
        self.num_layers = 3
        self.n_out_class = 1  # Binary classification, so output size is 1
        self.n_feat = 20
        self.criterion = nn.BCEWithLogitsLoss()

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.fc2 = nn.Linear(512, self.n_out_class)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def loss(self, y_pred, y_gt):
        return self.criterion(y_pred, y_gt)
