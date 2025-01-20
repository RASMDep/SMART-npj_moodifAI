import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

def get_data(args):
    
    """
    Load and process the dataset for training, validation, and testing. 
    Handles data partitioning, class weight calculation, and subject-specific exclusions.
    
    Parameters:
        args (Namespace): Argument namespace with paths, parameters, and configurations.

    Returns:
        tuple: Dataset objects for training, validation, and testing, along with class weights.
    """

    # Read dataset
    data_dir = args.data_path 
    data_file =args.data_file 
    fold_test = args.fold_test
    n_kfold = args.n_kfold
    target = args.target

    data = pd.read_pickle(os.path.join(data_dir,data_file))

    if args.test_subject != "none":

        data_all = pd.DataFrame(data.copy()).transpose()
        data_all = data_all[data_all.Participant_ID != args.test_subject]
        data_all = data_all.reset_index(drop=True)
        data_all = data_all[data_all['HR_Data'].apply(len) >= 288]  # Filter rows with insufficient data

        # Set up cross-validation splits
        cv = StratifiedKFold(n_splits=n_kfold, random_state=48, shuffle=True)
        yy = np.vstack(data_all[target][:])
        class_weights = compute_class_weight('balanced', classes=[0, 1, 2], y=data_all[target][:])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        train_idx, valid_idx = list(cv.split(data_all[target], yy))[fold_test]

        # Prepare datasets
        dataset_train = patchDataset(data_all, args, train_idx)
        dataset_val = patchDataset(data_all, args, valid_idx)

        # Prepare test dataset
        data_test = data_all[data_all.Participant_ID == args.test_subject].reset_index(drop=True)
        dataset_test = patchDataset(data_test, args, data_test.index)

    else:
        
        data_all = pd.DataFrame(data.copy()).transpose().reset_index(drop=True)
        data_all = data_all[data_all['HR_Data'].apply(len) >= 288]  # Filter rows with insufficient data

        # Set up cross-validation splits
        cv = StratifiedKFold(n_splits=n_kfold, random_state=48, shuffle=True)
        yy = np.vstack(data_all[target][:])
        class_weights = compute_class_weight('balanced', classes=[0, 1, 2], y=data_all[target][:])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        train_idx1, test_idx = list(cv.split(data_all[target], yy))[fold_test]
        train_idx, valid_idx = train_idx1[int(len(train_idx1) / 10):], train_idx1[:int(len(train_idx1) / 10)]

        # Prepare datasets
        dataset_train = patchDataset(data_all, args, train_idx)
        dataset_val = patchDataset(data_all, args, valid_idx)
        dataset_test = patchDataset(data_all, args, test_idx)

    return dataset_train, dataset_val,dataset_test,class_weights_tensor


class patchDataset(torch.utils.data.Dataset):

    """
    Dataset class that processes and returns HR data and associated features for training/testing.

    Attributes:
        n_channels (int): Number of channels in the input data.
        num_class (int): Number of target classes.
        x_data (pd.Series): HR data.
        y_data (pd.Series): Target labels.
        ...
    """

    def __init__(self,dataset,args, selected_idxs):

        self.num_class = args.num_class
        self.n_channels = args.n_channels
        self.x_data = dataset.loc[selected_idxs, 'HR_Data'].reset_index(drop=True)
        self.pid = dataset.loc[selected_idxs, 'Participant_ID']
        self.time = dataset.loc[selected_idxs, 'Date']
        self.y_data = dataset.loc[selected_idxs, args.target].reset_index(drop=True)
        
        # Other HRV and activity-related data
        self.activity_counts = dataset.loc[selected_idxs, 'activity_counts']
        self.step_count = dataset.loc[selected_idxs, 'step_count']
        self.resp_rate = dataset.loc[selected_idxs, 'resp_rate']
        self.depression = dataset.loc[selected_idxs, 'depression']
        self.sex = dataset.loc[selected_idxs, 'male']
        self.hour = args.hour
        self.combs = args.combs


    def __getitem__(self, index):
        """
        Retrieve a data sample by index, process it, and return the features and label.

        Parameters:
            index (int): Index of the sample to fetch.

        Returns:
            dict: Contains the features, labels, covariates, and metadata.
        """
        # Data transformation and normalization
        data_dict = self._get_normalized_data(index)
        
        # Build the final tensor
        x = self._build_input_tensor(data_dict)

        # Target label (one-hot encoded)
        y = torch.zeros(self.num_class)
        y[int(self.y_data[index])] = 1

        # Covariates (e.g., sex and depression)
        covariates = torch.tensor([self.sex[index], self.depression[index]], dtype=torch.float32)

        # Prepare the final sample
        sample = {
            'x': x,
            'label': y,
            'covariates': covariates,
            'pid': self.pid[index],
            'date': self.time[index],
            'mask': torch.ones((1, 1, 1)),
            'inputs': {'x': x, 'covariates': covariates},
        }

        return sample

    def __len__(self):
        return len(self.x_data)

    def _get_normalized_data(self, index):
        """
        Helper function to retrieve and normalize data features.

        Parameters:
            index (int): Index of the sample to fetch.

        Returns:
            dict: Normalized feature data for HR, activity, RMSSD, etc.
        """
        data_dict = {}
        
        # Feature extraction and normalization (e.g., HR, activity, RMSSD, etc.)
        data_dict['hr'] = self._normalize(self.x_data[index], min_val=33, max_val=180)[self.hour * 12:]
        data_dict['act'] = self._normalize(self.activity_counts[index], min_val=0, max_val=1000)[self.hour * 12:]
        data_dict['rmssd'] = self._normalize(self.RMSSD_Data[index], min_val=5, max_val=300)[self.hour * 12:]
        data_dict['en'] = self._normalize(self.SampEn_Data[index], min_val=0, max_val=2)[self.hour * 12:]
        data_dict['pnn50'] = self._normalize(self.pNN50_Data[index], min_val=0, max_val=100)[self.hour * 12:]
        data_dict['rr'] = self._normalize(self.resp_rate[index], min_val=12, max_val=25)[self.hour * 12:]
        data_dict['lfhf'] = self._normalize(np.log(self.LFHF_Data[index] + 1), min_val=2, max_val=5)[self.hour * 12:]
        data_dict['shan'] = self._normalize(self.ShanEn_Data[index], min_val=4, max_val=6)[self.hour * 12:]
        data_dict['hti'] = self._normalize(self.hti_Data[index], min_val=10, max_val=20)[self.hour * 12:]
        
        return data_dict

    def _normalize(self, data, min_val, max_val):
        """Normalize data to the range [0, 1] based on provided min and max values."""
        return (np.asarray(data) - min_val) / (max_val - min_val)

    def _build_input_tensor(self, data_dict):
        """Build the input tensor for the model based on the available channels."""
        x = np.zeros((self.n_channels, 288))  # Assuming the length is 288 (time steps)

        for i, feature in enumerate(data_dict):
            if self.n_channels > i:
                x[i, :len(data_dict[feature])] = data_dict[feature]
        
        return torch.tensor(x, dtype=torch.float32)