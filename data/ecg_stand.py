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

    dir = "/run/user/1000/gvfs/smb-share:server=hest.nas.ethz.ch,share=green_groups_sms_public/Projects_Current/Ambizione/01_Confidential_Data/MoodDetection/"
    #dir = "/home/gdapoian/Ambizione/01_Confidential_Data/MoodDetection/"
    # try with data on a pickle file, advantage load once
    # open the pickle file in read mode
    with open(os.path.join(dir,"4hlongecg_longacc_24hr_24hgps_and_imq.pickle"), "rb") as f:
        # load the data from the pickle file using pickle.load()
        data = pickle.load(f)

    subject = args.test_subject#"S008"

    data_training = {}
    data_training["x"] = data["x"][data["uid"]!=subject].reset_index( drop=True)#[data["uid"]!=subject].reset_index( drop=True) #[data["uid"]!=subject].reset_index( drop=True) #[np.logical_and(data["uid"]!=subject, data["night"]!=night)].reset_index( drop=True) #
    data_training["acc_x"] = data["acc_x"][data["uid"]!=subject].reset_index( drop=True)#[data["uid"]!=subject].reset_index( drop=True) #[np.logical_and(data["uid"]!=subject, data["night"]!=night)].reset_index( drop=True)[data["uid"]!=subject].reset_index( drop=True)
    data_training["acc_y"] = data["acc_y"][data["uid"]!=subject].reset_index( drop=True)#[data["uid"]!=subject].reset_index( drop=True) #[np.logical_and(data["uid"]!=subject, data["night"]!=night)].reset_index( drop=True)[data["uid"]!=subject].reset_index( drop=True)
    data_training["acc_z"] = data["acc_z"][data["uid"]!=subject].reset_index( drop=True)#[data["uid"]!=subject].reset_index( drop=True) #[np.logical_and(data["uid"]!=subject, data["night"]!=night)].reset_index( drop=True)[data["uid"]!=subject].reset_index( drop=True)  
    data_training["gps_lat"] = data["gps_lat"][data["uid"]!=subject].reset_index( drop=True)
    data_training["gps_long"] = data["gps_long"][data["uid"]!=subject].reset_index( drop=True)
    data_training["gps_time"] = data["gps_time"][data["uid"]!=subject].reset_index( drop=True)
    data_training["y"] = data["y"][data["uid"]!=subject].reset_index( drop=True)#[data["uid"]!=subject].reset_index(drop=True)
    data_training["daypart"] = data["daypart"][data["uid"]!=subject].reset_index( drop=True)#[data["uid"]!=subject].reset_index(drop=True)


    cv = StratifiedKFold(n_splits=n_kfold, random_state = 12, shuffle =True)
    yy = np.vstack(data_training["y"][:])[:,2]
    train_idx, valid_idx = list(cv.split(data_training['x'],yy))[fold_test]
    
    data_test = {}
    data_test["x"] = data["x"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True) #[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)
    data_test["acc_x"] = data["acc_x"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True)#[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)
    data_test["acc_y"] = data["acc_y"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True)#[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)
    data_test["acc_z"] = data["acc_z"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True)#[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)
    data_test["gps_lat"] = data["gps_lat"][data["uid"]==subject].reset_index( drop=True)
    data_test["gps_long"] = data["gps_long"][data["uid"]==subject].reset_index( drop=True)
    data_test["y"] = data["y"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True)#[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)
    data_test["daypart"] = data["daypart"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True)#[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)
    data_test["gps_time"] = data["gps_time"][data["uid"]==subject].reset_index( drop=True)#[data["uid"]==subject].reset_index( drop=True)#[np.logical_and(data["uid"]==subject, data["night"]==night)].reset_index( drop=True)

    test_idx =  np.asarray(data_test["y"].index)#


    dataset_train = ecgDataset(data_training, args, train_idx)  # create your datset
    dataset_val = ecgDataset(data_training, args, valid_idx) # create your datset
    dataset_test = ecgDataset(data_test, args, test_idx) # create your datset

    return dataset_train, dataset_val,dataset_test


class ecgDataset(torch.utils.data.Dataset):

    def __init__(self,files,args, select_idxs, mod=False):

    
        self.x_data = files['x'].copy()
        self.acc_x_data = files['acc_x'].copy()
        self.acc_y_data = files['acc_y'].copy()
        self.acc_z_data = files['acc_z'].copy()
        self.gps_lat_data = files['gps_lat'].copy()
        self.gps_long_data = files['gps_long'].copy()
        self.gps_time = files['gps_time'].copy()
        self.y_data = files['y'].copy()
        self.daypart = files['daypart'].copy()

    def __getitem__(self, index):

        hours = 8   
        len_ecg = 128*3600*hours
        len_acc = 5*3600*hours

        # get only 1 last hours of data before questionnaire (1382400 samples)
        x1 = np.zeros((4,len_ecg),dtype=float)
        x1[0,0:len(self.x_data[index])] = self.x_data[index][-len_ecg:]
        x1[0,:] = signal.filtfilt(b, a, x1[0,:]) # baseline removal is not improving
        x1[1,:] = signal.resample(self.acc_x_data[index][-len_acc:],len_ecg)
        x1[2,:] = signal.resample(self.acc_y_data[index][-len_acc:],len_ecg)
        x1[3,:] = signal.resample(self.acc_z_data[index][-len_acc:],len_ecg)
        x1 = (x1-x1.mean(axis=1, keepdims=True))
        x1= x1/(1e-12+x1.std(axis=1, keepdims=True))
        # data agumentation
        if random.random()<0.5:
            x1[0,:] = -x1[0,:]
        
        x2 = np.zeros((3,max([len(a) for a in self.gps_lat_data])))
        x2[0,0:len(self.gps_lat_data[index])]=self.gps_lat_data[index]
        x2[1,0:len(self.gps_long_data[index])]=self.gps_long_data[index]
        x2[2,0:len(self.gps_time[index])]=self.gps_time[index]
                       

        #AROUSAL
        arousal = np.zeros((3),dtype=float)
        #if self.y_data[index][0] >= 4 and np.abs(self.y_data[index][2]-6)<=2:
            #self.y_data[index][2] = np.abs(self.y_data[index][2]-6)
            #print("wrong label")
        arval = (self.y_data[index][0] + np.abs(self.y_data[index][2]-6))/2
       
        if arval<2:
            arousal[0]=1
        elif arval>4:
            arousal[2]=1
        else:
            arousal[1]=1

        #VALENCE
        valence = np.zeros((3),dtype=float)
        vaval = (self.y_data[index][1] + np.abs(self.y_data[index][3]-6))/2
        
        if vaval<2:
            valence[0]=1
        elif vaval>4:
            valence[2]=1
        else:
            valence[1]=1

        #time of the day
        daypart = np.zeros((3),dtype=float)
        if self.daypart[index]==0:
            daypart[0]=1
        elif self.daypart[index]==1:
            daypart[1]=1
        elif self.daypart[index]==2:
            daypart[2]=1
        
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        arousal = torch.from_numpy(arousal).float()
        valence = torch.from_numpy(valence).float()
        daypart = torch.from_numpy(daypart).float()

        mask = torch.ones((1,1,1))

        covariates = torch.ones((1,1,1))
        hea_file_name = " "

  
        sample = {
            'x1': x1,
            'x2': x2,
            'label_arosual':arousal,
            'label_valence':valence,
            'label_daypart':daypart,
            'covariates':0,
            'name': hea_file_name,
            'mask': mask,
            'was_changed': 0,
        }

        return sample

    def __len__(self):

        return self.x_data.shape[0]