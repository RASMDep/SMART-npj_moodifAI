import numpy as np
import torch
from models import model_GRU, model_Conv1dNet, model_Conv1dNetSegments, model_LSTM, model_TCN, model_ResNet, model_twoConv1dNet_MLP 

def get_model(args,data=None):

    # Define model
    if args.model == 'Conv1dNet': 
        print(args.model)
        model = model_Conv1dNet.Conv1dNet(args)
    elif args.model == 'twoConv1dNet_MLP': 
        model = model_twoConv1dNet_MLP.twoConv1dNet_MLP(args)
    else:
        raise NotImplementedError
    
    return model
