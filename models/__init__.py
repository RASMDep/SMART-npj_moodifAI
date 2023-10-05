import numpy as np
import torch
from models import model_GRU, model_Conv1dNet, model_Conv1dNetSegments, model_LSTM, model_TCN, model_ResNet, model_twoConv1dNet_MLP 

def get_model(args,data=None):

    # Define model
    if args.model == 'Conv1dNet': 
        print(args.model)
        model = model_Conv1dNet.Conv1dNet(args)
    elif args.model == 'Conv1dNetSegments': 
        model = model_Conv1dNetSegments.Conv1dNetSegments(args)
    elif args.model == 'twoConv1dNet_MLP': 
        model = model_twoConv1dNet_MLP.twoConv1dNet_MLP(args)
    elif args.model == 'GRU': 
        model = model_GRU.GRU(args)
    elif args.model == 'LSTM': 
        model = model_LSTM.LSTM(args)
    elif args.model == 'TCN': 
        model = model_TCN.TCN(args)
    elif args.model == 'ResNet1d': 
        model = model_ResNet.resnet34(input_channels=12)
    else:
        raise NotImplementedError
    
    return model
