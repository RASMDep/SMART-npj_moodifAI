import numpy as np
import data.patch_data 

def get_data(args,data_stats=None):

    if args.data == "patch_data":
        return patch_data.get_data(args)
    else:
        raise NotImplementedError