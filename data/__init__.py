import numpy as np
import data.ecg_stand 

def get_data(args,data_stats=None):

    if args.data == "ecg_data":
        return ecg_stand.get_data(args)
    else:
        raise NotImplementedError