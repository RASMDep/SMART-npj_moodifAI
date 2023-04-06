import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import io

def new_log(base_path,base_name,style=True,args=None):
    name = base_name

    folder_path = os.path.join(base_path) #,args.data)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    folder_path = os.path.join(folder_path) #,base_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    previous_runs = os.listdir(folder_path)
    n_exp = len(previous_runs)

    #experiment_folder = os.path.join(folder_path,"experiment_{}".format(n_exp))
    experiment_folder = folder_path
    
    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        with open(os.path.join(experiment_folder, "args" + '.txt'), 'w') as f:
            sorted_names = sorted(args_dict.keys(), key=lambda x: x.lower())
            for key in sorted_names:
                value = args_dict[key]
                f.write('%s:%s\n' % (key, value))

    return experiment_folder

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=4, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.track_lc_progress = True
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            if self.counter > 0:
                self.counter = 0
                self.track_lc_progress = True
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.track_lc_progress = False
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True