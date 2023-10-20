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

b, a = signal.butter(4, 0.25, 'highpass', fs=128)
imq = 0


def fix_baseline_wander(data, fs=128):
    """BaselineWanderRemovalMedian.m from ecg-kit.  Given a list of amplitude values
    (data) and sample rate (sr), it applies two median filters to data to
    compute the baseline.  The returned result is the original data minus this
    computed baseline.
    """
    #source : https://pypi.python.org/pypi/BaselineWanderRemoval/2017.10.25

    winsize = int(round(0.2*fs))
    # delayBLR = round((winsize-1)/2)
    if winsize % 2 == 0:
        winsize += 1
    baseline_estimate = medfilt(data, kernel_size=winsize)
    winsize = int(round(0.6*fs))
    # delayBLR = delayBLR + round((winsize-1)/2)
    if winsize % 2 == 0:
        winsize += 1
    baseline_estimate = medfilt(baseline_estimate, kernel_size=winsize)
    ecg_blr = data - baseline_estimate
    return ecg_blr


def get_data(args):

    # Read dataset
    dir = args.data_path
    fold_test = args.fold_test
    n_kfold = args.n_kfold

    subject = "SMART_012"
    val_subject = "SMART_012"

    data_dir = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/read_only/SMART_derived_features/"
    # Specify the path to the pickle file you want to read
    file_path = "HRV_ACC_timeseries_24hours_clean.pkl"  # Replace with the actual file path


    # Open the pickle file for reading in binary mode
    with open(os.path.join(data_dir,file_path), 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)

    data_training = {key[1]:value for key, value in data.items() if not key[0].startswith(subject) or key[0].startswith(val_subject) }
    data_training = pd.DataFrame(data_training).transpose()
    train_idx =  np.asarray(data_training["imq2"].index)


    data_validation = {key[1]:value for key, value in data.items() if key[0].startswith(val_subject)}
    data_validation = pd.DataFrame(data_validation).transpose()
    valid_idx =  np.asarray(data_validation["imq2"].index)

    
    data_test = {key[1]:value for key, value in data.items() if key[0].startswith(subject)}
    data_test = pd.DataFrame(data_test).transpose()

    test_idx =  np.asarray(data_test["imq2"].index)

    dataset_train = ecgDataset(data_training, args, train_idx)  # create your datset
    dataset_val = ecgDataset(data_validation, args, valid_idx) # create your datset
    dataset_test = ecgDataset(data_test, args, test_idx) # create your datset


    return dataset_train, dataset_val,dataset_test


class ecgDataset(torch.utils.data.Dataset):

    def __init__(self,dataset,args, selected_idxs, mod=False):

        data = dataset.loc[selected_idxs]
        data = data.reset_index()
        self.x_data = data['HR_Data'].copy()
        self.RMSSD_Data = data['RMSSD_Data'].copy()
        self.SampEn_Data = data['SampEn_Data'].copy()
        self.activity_counts = data['activity_counts'].copy()
        self.step_count = data['step_count'].copy()
        self.run_walk_time = data['run_walk_time'].copy()
        self.y_data = data['imq2'].copy()
        self.y_data =  self.y_data>3

    def __getitem__(self, index):


        x1 = np.asarray(self.x_data[index])
        x1 = (x1 - 33) / (180 - 33)

        x2 = np.asarray(self.step_count[index])
        x2 = x2/100

        x3 = np.asarray(self.SampEn_Data[index])

        x=np.zeros((2,288))
        x[0,0:len(x1)] = x1
        x[1,0:len(x2)] = x3  
        #x[2,0:len(x2)] = x3  
        x = np.nan_to_num(x, nan=0)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(np.asarray(self.y_data[index])).float()

        mask = torch.ones((1,1,1))

        covariates = torch.ones((1,1,1))
        hea_file_name = " "

  
        sample = {
            'x': x,
            #'x2': x2,
            'label':y,
            'covariates':0,
            'name': hea_file_name,
            'mask': mask,
            'was_changed': 0,
        }

        return sample

    def __len__(self):

        return self.x_data.shape[0]
    