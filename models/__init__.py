import numpy as np
import torch
from models import model_Conv1dNet, lstm

def get_model(args,data=None):

    # Define model
    if args.model == 'Conv1dNet': 
        print(args.model)
        model = model_Conv1dNet.Conv1dNet(args)
    else:
        raise NotImplementedError
    
    return model
