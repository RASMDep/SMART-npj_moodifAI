import random
import numpy as np
import torch
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
from random import sample, shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
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

ids_p = ['SMART_201', 'SMART_003', 'SMART_004','SMART_006', 'SMART_007','SMART_008', 'SMART_009','SMART_010', 
        'SMART_012','SMART_015', 'SMART_016','SMART_018','SMART_024',
        'SMART_019','SMART_020','SMART_027','SMART_028']


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

        dataset_train = patchDataset(data_all, args, train_idx)  # create your dataset
        dataset_val = patchDataset(data_all, args, valid_idx) # create your dataset
        dataset_test = patchDataset(data_test, args, test_idx) # create your dataset

    else:
        
        data = pd.read_pickle(os.path.join(data_dir,data_file))

        data_all = pd.DataFrame(data.copy()).transpose()
        data_all = data_all.reset_index()
        data_all = data_all.drop([i for i in range(len(data_all)) if len(data_all.loc[i, "HR_Data"]) < 288])
        data_all = data_all.reset_index(drop=True)  # Use drop=True to remove the old index column

        cv = StratifiedKFold(n_splits=n_kfold, random_state = 48, shuffle =True)
        yy = np.vstack(data_all[target][:])
        class_weights = compute_class_weight('balanced',classes=[0,1,2],y=data_all[target][:])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        train_idx1, test_idx = list(cv.split(data_all[target],yy))[fold_test]
  
        train_idx = train_idx1[int(len(train_idx1)/10):]
        valid_idx = train_idx1[0:int(len(train_idx1)/10)]

        dataset_train = patchDataset(data_all, args, train_idx)  # create your dataset
        dataset_val = patchDataset(data_all, args, valid_idx) # create your dataset
        dataset_test = patchDataset(data_all, args, test_idx) # create your dataset

    return dataset_train, dataset_val,dataset_test,class_weights_tensor


class patchDataset(torch.utils.data.Dataset):

    def __init__(self,dataset,args, selected_idxs, mod=False):

        self.num_class = args.num_class
        self.n_channels = args.n_channels
        data = dataset.loc[selected_idxs]
        data = data.reset_index()
        self.x_data = data['HR_Data'].copy()
        self.minute = data['Minute'].copy()
        self.RMSSD_Data = data['RMSSD_Data'].copy()
        self.SampEn_Data = data['SampEn_Data'].copy()
        self.activity_counts = data['activity_counts'].copy()
        self.step_count = data['step_count'].copy()
        self.resp_rate = data['resp_rate'].copy()
        self.run_walk_time = data['run_walk_time'].copy()
        self.y_data = data[args.target].copy()
        self.day_part = data['quest_type'].copy()
        self.depression = data['depression'].copy()
        self.sex = data['male'].copy()
        self.hour = args.hour
        self.combs = args.combs


    def __getitem__(self, index):

        x1 = np.asarray(self.x_data[index])
        x1 = (x1 - 33) / (180 - 33)
        hr = x1[self.hour*12:]

        x2 = np.asarray(self.activity_counts[index])
        x2 = (x2 - 0) / (1000-0)
        act = x2[self.hour*12:]

        x3 = np.asarray(self.RMSSD_Data[index])
        x3 = (x3 - 5) / (300 - 5)
        rmssd = x3[self.hour*12:]

        x4 = np.asarray(self.SampEn_Data[index])
        x4 = (x4 - np.nanmin(x4)) / (np.nanmax(x4)-np.nanmin(x4))
        x4 = (x4 - 0) / (1-0)
        en = x4[self.hour*12:]

        x5 = np.asarray(self.resp_rate[index])
        x5 = (x5 - 12) / (25-12)
        rr = x5[self.hour*12:]

        #x6 = np.asarray(self.minute[index])
        #x6 = (x6 - 0) / (1440 - 0)

        mod_dict = {'hr':hr,'act':act,'rmssd':rmssd,'en':en,'rr':rr}
 
        x=np.zeros((self.n_channels,288))

        if self.n_channels==1:

            x1 = mod_dict[self.combs]
            
            x[0,0:len(x1)] = x1
        
        elif self.n_channels==2:

            x1 = mod_dict[self.combs.split('+')[0]]
            x2 = mod_dict[self.combs.split('+')[1]]

            x[0,0:len(x1)] = x1
            x[1,0:len(x2)] = x2
        
        elif self.n_channels==3:

            x1 = mod_dict[self.combs.split('+')[0]]
            x2 = mod_dict[self.combs.split('+')[1]]
            x3 = mod_dict[self.combs.split('+')[2]]

            x[0,0:len(x1)] = x1
            x[1,0:len(x2)] = x2
            x[2,0:len(x3)] = x3
        
        elif self.n_channels==4:

            x1 = mod_dict[self.combs.split('+')[0]]
            x2 = mod_dict[self.combs.split('+')[1]]
            x3 = mod_dict[self.combs.split('+')[2]]
            x4 = mod_dict[self.combs.split('+')[3]]

            x[0,0:len(x1)] = x1
            x[1,0:len(x2)] = x2
            x[2,0:len(x3)] = x3
            x[3,0:len(x4)] = x4
        
        elif self.n_channels==5:

            x1 = mod_dict[self.combs.split('+')[0]]
            x2 = mod_dict[self.combs.split('+')[1]]
            x3 = mod_dict[self.combs.split('+')[2]]
            x4 = mod_dict[self.combs.split('+')[3]]
            x5 = mod_dict[self.combs.split('+')[4]]

            x[0,0:len(x1)] = x1
            x[1,0:len(x2)] = x2
            x[2,0:len(x3)] = x3
            x[3,0:len(x4)] = x4
            x[4,0:len(x5)] = x5
        x[0,0:len(x1)] = x1
        x[1,0:len(x4)] = x4
        x[2,0:len(x3)] = x3
        x[3,0:len(x5)] = x5
        x[4,0:len(x2)] = x2

        x = np.nan_to_num(x, nan=0)

        x = torch.from_numpy(x).float()
        y = torch.zeros((self.num_class))
        y[int(self.y_data[index])] = 1

        mask = torch.ones((1,1,1))

        covariates = torch.ones((1,1,1))
        #covariates =  torch.tensor([self.sex[index],self.depression[index]]).view(2)
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
    