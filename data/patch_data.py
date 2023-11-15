import random
import numpy as np
import torch
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
from random import sample, shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
from scipy import signal
from scipy.signal import medfilt
import datetime
import numpy as np


# Dictionary mapping string values to numeric values
mapping = {
    'MORNING_INTERVENTION': 0,
    'morning': 0,
    'MORNING': 0,
    'kss_mood': 0.5,
    'evening': 1,
    'EVENING': 1
}

ids_p = ['SMART_201', 'SMART_001', 'SMART_003', 'SMART_004', 'SMART_006', 'SMART_008', 'SMART_009', 'SMART_007', 'SMART_010', 'SMART_012', 'SMART_015', 'SMART_016', 'SMART_018', 'SMART_019']


def get_data(args):

    # Read dataset
    data_dir = args.data_path 
    data_file =args.data_file  # Replace with the actual file path
    fold_test = args.fold_test
    n_kfold = args.n_kfold
    target = args.target

    if args.test_subject != "none":

        data = pd.read_pickle(os.path.join(data_dir,data_file))

        data_all = pd.DataFrame(data.copy()).transpose()
        data_all = data_all[data_all.Participant_ID!=args.test_subject]
        data_all = data_all.reset_index()
        data_all = data_all.drop([ i for i in range(0,len(data_all)) if len(data_all.loc[i,"HR_Data"])<288])
        data_all = data_all.reset_index(drop=True)

        cv = StratifiedKFold(n_splits=n_kfold, random_state = 48, shuffle =True)
        yy = np.vstack(data_all[target][:])
        train_idx, valid_idx = list(cv.split(data_all[target],yy))[fold_test]

        data_test = pd.DataFrame(data.copy()).transpose()
        data_test = data_test[data_test.Participant_ID==args.test_subject]
        data_test = data_test.reset_index()
        test_idx = data_test.index

        dataset_train = patchDataset(data_all, args, train_idx)  # create your datset
        dataset_val = patchDataset(data_all, args, valid_idx) # create your datset
        dataset_test = patchDataset(data_test, args, test_idx) # create your datset

    else:
        
        data = pd.read_pickle(os.path.join(data_dir,data_file))

        data_all = pd.DataFrame(data.copy()).transpose()
        data_all = data_all.reset_index()
        data_all = data_all.drop([i for i in range(len(data_all)) if len(data_all.loc[i, "HR_Data"]) < 288])
        data_all = data_all.reset_index(drop=True)  # Use drop=True to remove the old index column


        cv = StratifiedKFold(n_splits=n_kfold, random_state = 48, shuffle =True)
        yy = np.vstack(data_all[target][:])
        train_idx1, test_idx = list(cv.split(data_all[target],yy))[fold_test]
        train_idx = train_idx1[int(len(train_idx1)/10):]
        valid_idx = train_idx1[0:int(len(train_idx1)/10)]


        dataset_train = patchDataset(data_all, args, train_idx)  # create your datset
        dataset_val = patchDataset(data_all, args, valid_idx) # create your datset
        dataset_test = patchDataset(data_all, args, test_idx) # create your datset


    return dataset_train, dataset_val,dataset_test


class patchDataset(torch.utils.data.Dataset):

    def __init__(self,dataset,args, selected_idxs, mod=False):

        self.num_class = args.num_class
        data = dataset.loc[selected_idxs]
        data = data.reset_index()
        self.x_data = data['HR_Data'].copy()
        self.RMSSD_Data = data['RMSSD_Data'].copy()
        self.SampEn_Data = data['SampEn_Data'].copy()
        self.activity_counts = data['activity_counts'].copy()
        self.step_count = data['step_count'].copy()
        self.run_walk_time = data['run_walk_time'].copy()
        self.y_data = data[args.target].copy()
        self.day_part = data['quest_type'].copy()
        self.depression = data['depression'].copy()


    def __getitem__(self, index):

        x1 = np.asarray(self.x_data[index])
        #x1 = (x1 - 33) / (180 - 33)
        x1 = (x1 - np.nanmin(x1)) / (np.nanmax(x1)-np.nanmin(x1))

        x2 = np.asarray(self.activity_counts[index])
        x2 = (x2 - np.nanmin(x2)) / (np.nanmax(x2)-np.nanmin(x2))

        x=np.zeros((2,288))
        x[0,0:len(x1)] = x1
        x[1,0:len(x2)] = x2

        x = np.nan_to_num(x, nan=0)

        x = torch.from_numpy(x).float()
        y = torch.zeros((self.num_class))
        y[int(self.y_data[index])] = 1

        mask = torch.ones((1,1,1))

        covariates = torch.ones((1,1,1))
        #covariates =  torch.tensor( self.depession[index]).view(1)
        hea_file_name = " "

  
        sample = {
            'x': x,
            'label':y,
            'covariates':covariates,
            'name': hea_file_name,
            'mask': mask,
            'was_changed': 0,
        }

        return sample

    def __len__(self):

        return self.x_data.shape[0]
    