# Import libraries
import sys, argparse, os
import numpy as np
from numpy.core.fromnumeric import squeeze
import scipy
import torch
import csv
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import math
import pickle
import itertools

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from dev.selfLC import selfLC

from utils.utils import *
from data import get_data
from models import get_model
from models import *
import sklearn.metrics as metrics
from torch.utils.data import WeightedRandomSampler

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm, trange

parser = argparse.ArgumentParser()

### general parameters ################################################
parser.add_argument(
    '--mode', default="test_train",
    type=str, choices=["test","train"],help="mode to be run")
parser.add_argument("--per-change", default=0, type=int)
parser.add_argument("--save-dir", type=str, help="folder to save results")
parser.add_argument("--model-filename",
                    type=str,
                    help="name to use to save model")
parser.add_argument("--save-model",
                    default="last",
                    choices=['last', 'best', 'No'],
                    help="which model to use for testing and saving")
parser.add_argument('--resume',
                    type=str,
                    default=None,
                    help='put the path to resuming file if needed')
parser.add_argument("--root",
                    default="/Users/giulia/Documents/ECG_challenge",
                    type=str,
                    help="root folder")
parser.add_argument('--tag', default="__", type=str)
parser.add_argument("--val-every-n-epochs",
                    type=int,
                    default=1,
                    help="interval of training epochs to run the validation")
parser.add_argument("--experiment-folder",
                    default="/Users/giulia/Documents/PhysioNetChallenge2021_SMS/",
                    type=str,
                    help="experiment folder")

### k-fold cross validation ########################################
parser.add_argument("--fold-test",
                    type=int,
                    default=0,
                    help="k fold run used for testing")
parser.add_argument("--test-subject",
                    type=str,
                    default=0,
                    help="subject to leav out for test")
parser.add_argument("--n-kfold", type=int, default=5, help="How many folds?")

### data parameters ################################################
parser.add_argument("--data",
                    default="mnist",
                    type=str,
                    choices=["ecg_data", "mnist"],
                    help="dataset selection")
parser.add_argument("--data-path",
                    default="/scratch/tmp/full_train_ROIZ_2016_summer.pkl",
                    type=str,
                    help="location of the data")
parser.add_argument("--resamp-freq",
                    type=int,
                    default=125,
                    help="resample the data to this frequency")
parser.add_argument(
    "--save-covariates",
    default=False,
    action="store_true",
    help="only compute covariates once, and if they are computed reuse them")
parser.add_argument("--recompute-covariates",
                    default=False,
                    action="store_true",
                    help="if set, covariates are recomputed")
parser.add_argument("--covariates-save-path",
                    default="features/",
                    type=str,
                    help="location to store the covariates")
parser.add_argument(
    "--num-processes",
    type=int,
    default=1,
    help='number of processes used to compute features [default=%(default)s].')

### optimizer parameters ################################################
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help='Maximum number of epochs [default=%(default)s].')
parser.add_argument("--logstep-train",
                    default=50,
                    type=int,
                    help="iterations step for training log")
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help='Optimizer initial learning rate')
parser.add_argument("--momentum",
                    type=float,
                    default=0.9,
                    help='momentum for sgd')
parser.add_argument(
    "--batch-size",
    type=int,
    default=600,
    help='Batch size for train, validation and test sets [default=%(default)s].'
)
parser.add_argument("--num-workers",
                    type=int,
                    default=0,
                    help='Num workers for dataloaders [default=%(default)s].')
parser.add_argument("--scheduler",
                    default="no",
                    choices=['no', 'exp'],
                    help='add scheduler default=no')
parser.add_argument("--scheduler-decay-factor",
                    type=float,
                    default=0.9,
                    help='Learning rate decay factor [default=%(default)s].')
parser.add_argument("--scheduler-step",
                    type=str,
                    default="epoch",
                    choices=["epoch", "iter"],
                    help='how often a scheduler step is taken')
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.,
    help='Add L2 regularization to the loss function [default=%(default)s].')

### model parameters ################################################
parser.add_argument(
    "--model",
    type=str,
    default='LeNet',
    choices=['Conv1dNet', 'LeNet', 'Conv1dNet_10s', 'GRU', 'LSTM', 'TCN', 'ResNet1d', 'twoConv1dNet_MLP'],
    help='Model to use [default=%(default)s].')
parser.add_argument("--n-channels",
                    type=int,
                    default=12,
                    help="number of ECG channels")
parser.add_argument("--num-class",
                    type=int,
                    default=30,
                    help="number of classes")
parser.add_argument(
    "--n-covariates",
    type=int,
    default=0,
    help="number of additioanl manual features to add to last layer")
parser.add_argument("--use-covariates",
                    action="store_true",
                    default=False,
                    help="whether or not to use handcrafted features")
parser.add_argument("--loss",
                    type=str,
                    default='BCE',
                    choices=['BCE', 'ASL', 'challenge'],
                    help='Loos to use [default=%(default)s].')
parser.add_argument("--gamma-neg",
                    type=int,
                    default=3,
                    help='contribution of negative samples to ASL loss')
parser.add_argument("--gamma-pos",
                    type=int,
                    default=1,
                    help='contribution of positive samples to ASL loss')
parser.add_argument("--less-features",
                    default=False, action="store_true",
                    help='extra layes in ConvNet for fewer features output')

### bias parameters #################################################
parser.add_argument("--add-bias", default=False, action="store_true")
parser.add_argument("--noise", type=float, default=0.1)


### for GRU
### TO DO add hidden_channels, num_layers


class DevelopingSuite(object):
    def __init__(self, args):

        self.args = args

        self.data_train, self.data_val, self.data_test = get_data(args)

        # Define torch dataloader object

        #weight = 1. / np.array([len(np.where(self.data_train.y_data == t)[0]) for t in np.unique(self.data_train.y_data)])
        #print(np.array([len(np.where(self.data_train.y_data == t)[0]) for t in np.unique(self.data_train.y_data)]))
        #samples_weight = np.array([weight[int(t)] for t in self.data_train.y_data])
        #samples_weight = torch.from_numpy(samples_weight)
        #sampler = WeightedRandomSampler(samples_weight, len(self.data_train.y_data), replacement=True)

        self.train_dataloader = DataLoader(self.data_train,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           drop_last=True,
                                           shuffle=True)
        self.val_dataloader = DataLoader(self.data_val,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers)

        args.n_covariates = 0 

        # Use GPU if available
        print('GPU devices available:', torch.cuda.device_count())
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.model = get_model(args)
        self.model.to(self.device)

        self.experiment_folder = args.save_dir

        if args.resume is not None:

            #self.resume(path=args.resume)

            raise NotImplemented

        if "train" in args.mode:
            self.experiment_folder = new_log(args.save_dir,args.model + "_" + args.tag,args=args)
            if not os.path.isdir(self.experiment_folder):
                os.mkdir(self.experiment_folder)
            self.writer = SummaryWriter(log_dir=self.experiment_folder)

            if args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=args.lr,
                    weight_decay=self.args.weight_decay)
            elif args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay)

            if args.scheduler == "exp":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.args.scheduler_decay_factor)
                #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, factor=args.lr_decay_factor, mode='max', verbose=True)
            elif args.scheduler == "no":
                self.scheduler = None
            else:
                print("scheduler unknown")
                print(args.scheduler)

            self.epoch = 0
            self.iter = 0
            self.train_stats = {}
            self.val_stats = {}
            self.lastepoch = False

            self.early_stopping = EarlyStopping()  #initialize early stopping

    def train_and_eval(self):

        self.val_stats["validation_loss"] = torch.tensor(float('nan'))
        self.val_stats["best_validation_loss"] = torch.tensor(float('nan'))
        self.val_stats["validation_accuracy"] = torch.tensor(float('nan'))

        self.task = 1
        with tqdm(range(0, self.args.epochs), leave=True) as tnr:
        
            tnr.set_postfix(
                best_validation_loss=self.val_stats["best_validation_loss"],
                validation_loss=self.val_stats["validation_loss"],
                accuracy=self.val_stats["validation_accuracy"])

            for n, self.epoch in enumerate(tnr):
               
                self.training(tnr)

                if self.epoch % self.args.val_every_n_epochs == 0 and self.args.val_every_n_epochs != -1:
                    self.validate()

                    self.early_stopping(self.val_stats["validation_loss"])
                    if self.early_stopping.early_stop:
                        break

                if self.scheduler is not None and self.args.scheduler_step == "epoch":
                    self.scheduler.step()
                    self.writer.add_scalar(
                        'log_lr', np.log10(self.scheduler.get_lr()),
                        self.iter)

        if self.args.save_model == "best":
            self.save_model()


    def training(self, tnr=None):

        self.model.train()
        accumulated_loss, accumulated_main_loss, accumulated_func_eval = 0, 0, 0

        n_corr = 0
        l = 0

        lc_recordings = []
        lc_original_y = []
        lc_corrected_y = []

        with tqdm(self.train_dataloader, leave=False) as inner_tnr:
            for en, sample in enumerate(inner_tnr):

                y_arousal = sample['label_arosual'].to(self.device)
                y_valence = sample['label_valence'].to(self.device)
                y_daypart = sample['label_daypart'].to(self.device)

                self.optimizer.zero_grad()
                
                pred_arousal, pred_valence, pred_daypart = self.model(sample)#.squeeze()

                loss =  self.model.loss(pred_arousal,y_arousal) 
                + self.model.loss(pred_valence,y_valence) 
                + self.model.loss(pred_daypart,y_daypart) 
                
                # Backward pass
                loss.backward()
                self.optimizer.step()

                accumulated_loss += loss.item()

                self.iter += 1

                # adjust learning rate
                if self.scheduler is not None and self.args.scheduler_step == "iter":
                    self.scheduler.step()
                    self.writer.add_scalar('log_lr',
                                           np.log10(self.scheduler.get_lr()),
                                           self.iter)

                # log progress
                if (en + 1) % self.args.logstep_train == 0:
                    self.train_stats[
                        'train_loss'] = accumulated_loss / self.args.logstep_train
                    func_eval = accumulated_func_eval / (
                        self.args.logstep_train)
                    accumulated_loss, accumulated_func_eval = 0., 0.
                    inner_tnr.set_postfix(
                        training_loss=self.train_stats['train_loss'])
                    if tnr is not None:
                        tnr.set_postfix(
                            training_loss=self.train_stats['train_loss'],
                            best_validation_loss=self.
                            val_stats["best_validation_loss"],
                            validation_loss=self.val_stats["validation_loss"],
                            accuracy=self.val_stats["validation_accuracy"])

                    self.writer.add_scalar('training/training loss',
                                           self.train_stats['train_loss'],
                                           self.iter)
                    # TODO: this is not computed
                    #  self.writer.add_scalar('training/func_eval', func_eval,
                    #  self.iter)
        

    def validate(self, tnr=None, save=True):

        total_dataset_size = 0
        total_loss = 0
        total_accuracy = 0

        nclasses = self.args.num_class

        outputs_all = torch.zeros(0, nclasses).to(self.device)
        #labels_all = torch.zeros(0, nclasses).to(self.device)

        dataloader = self.val_dataloader
        self.model.eval()
        for sample in dataloader:
            with torch.no_grad():
                x1 = sample["x1"]
                arousal = sample["label_arosual"]
                valence = sample["label_valence"]
                daypart = sample["label_daypart"]

                arousal = arousal.to(self.device)
                valence = valence.to(self.device)
                daypart = daypart.to(self.device)
                batch_size = x1.size(0)
                pred_arousal,pred_valence,pred_daypart = self.model(sample)
                pred_arousal = pred_arousal.to(self.device) 
                pred_valence = pred_valence.to(self.device) 
                pred_daypart = pred_daypart.to(self.device) 

                #y_pred_binary_arousal = 0.5 * (torch.sign(torch.sigmoid(pred_arousal) - 0.5) + 1).to(self.device)
                #y_pred_binary_valence = 0.5 * (torch.sign(torch.sigmoid(pred_valence) - 0.5) + 1).to(self.device)

                #outputs_all = torch.cat((outputs_all, y_pred_binary_arousal,y_pred_binary_valence))
                #labels_all = torch.cat((labels_all, y.unsqueeze(1)))

                # Add metrics of current batch
                total_dataset_size += batch_size
                #total_loss += (self.model.loss(pred_arousal1,arousal1)  
                #+ 
                total_loss += (self.model.loss(pred_valence,valence) 
                +  self.model.loss(pred_daypart,daypart) 
                + self.model.loss(pred_arousal,arousal)  
                                )* batch_size #+  self.model.loss(pred_valence,valence))* batch_size

        total_accuracy += 0 #compute_total_matches(y, pred_y)

        # Average metrics over the whole dataset
        total_loss /= total_dataset_size
        total_accuracy /= total_dataset_size

        self.val_stats["validation_loss"] = total_loss.item()
        self.val_stats["validation_accuracy"] = 0  #total_accuracy.item()
 
        if not self.val_stats["best_validation_loss"] < self.val_stats[
                "validation_loss"]:
            self.val_stats["best_validation_loss"] = self.val_stats[
                "validation_loss"]
            if save and self.args.save_model == "best":
                self.save_model()

        self.writer.add_scalar('validation/validation loss',
                               self.val_stats["validation_loss"], self.epoch)
        self.writer.add_scalar('validation/best validation loss',
                               self.val_stats["best_validation_loss"],
                               self.epoch)
        self.writer.add_scalar('validation/accuracy',
                               self.val_stats["validation_accuracy"],
                               self.epoch)

        return



    def save_model(self):

        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'args': self.args
            }, os.path.join(self.experiment_folder, self.args.model_filename))
        #self.args.covariates = covariates_tmp

    def eval_model_stats(self):

        data_model = torch.load(
            os.path.join(self.experiment_folder, self.args.model_filename))

        self.model.load_state_dict(data_model['model_state_dict'])

        #evaluation
        self.model.eval()
        nclasses = self.args.num_class
        acc = torch.zeros(1).to(
            self.device
        )  # <-- changed so it the number of classes is automatic
        pos = torch.zeros(1).to(
            self.device)  # <-- changed so it is automatic

        outputs_all_arousal = torch.zeros(0).to(
            self.device)  # <-- changed so it is automatic
        outputs_all_valence = torch.zeros(0).to(
            self.device)  # <-- changed so it is automatic
        outputs_all_daypart = torch.zeros(0).to(
            self.device)  # <-- changed so it is automatic
      

        targets_all_arousal = torch.zeros(0).to(
            self.device)  # <-- changed so it is automatic
        targets_all_valence = torch.zeros(0).to(
                    self.device)  # <-- changed so it is automatic
        targets_all_daypart = torch.zeros(0).to(
                    self.device)  # <-- changed so it is automatic
        
        tot = 0

        self.args.mode = "test"
        #self.data_test,_  = get_data(self.args)
        # Define torch dataloader object
        self.test_dataloader = DataLoader(self.data_test,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers)


        with torch.no_grad():
            for sample in tqdm(self.test_dataloader):
                x_t = sample["x1"]
                y_t_arousal = sample["label_arosual"].to(self.device)
                y_t_valence = sample["label_valence"].to(self.device)
                y_t_daypart = sample["label_daypart"].to(self.device)
                
                tot = tot + x_t.shape[0]
                y_pred_arousal, y_pred_valence, y_pred_daypart = self.model(sample)
                y_pred_binary_arousal = torch.argmax(torch.sigmoid(y_pred_arousal), dim=1 ).float()#0.5 * (torch.sign(torch.sigmoid(y_pred) - 0.5) + 1)
                y_pred_binary_valence = torch.argmax(torch.sigmoid(y_pred_valence), dim=1 ).float()#0.5 * (torch.sign(torch.sigmoid(y_pred) - 0.5) + 1)
                y_pred_binary_daypart = torch.argmax(torch.sigmoid(y_pred_daypart), dim=1 ).float()#0.5 * (torch.sign(torch.sigmoid(y_pred) - 0.5) + 1)
        
                outputs_all_arousal = torch.cat((outputs_all_arousal, y_pred_binary_arousal), 0)
                outputs_all_valence = torch.cat((outputs_all_valence, y_pred_binary_valence), 0)
                outputs_all_daypart = torch.cat((outputs_all_daypart, y_pred_binary_daypart), 0)
        
                
                targets_all_arousal = torch.cat((targets_all_arousal, y_t_arousal,), 0)
                targets_all_valence = torch.cat((targets_all_valence, y_t_valence,), 0)
                targets_all_daypart = torch.cat((targets_all_daypart, y_t_daypart,), 0)

                #target = torch.argmax(torch.sigmoid(y_t_arosual), dim=1 ).float()
                #acc = acc + (torch.sum(y_pred_binary_arousal == target)) 
                #pos = pos + self.args.batch_size #torch.sum(y_t, dim=0)

        return outputs_all_arousal,outputs_all_valence,outputs_all_daypart,targets_all_arousal,targets_all_valence,targets_all_daypart


if __name__ == '__main__':

    args = parser.parse_args()

    # compute handcrafted features first
    if args.use_covariates:
        args.covariates = precompute_covariates(args)
    else:
        args.covariates = []

    developingSuite = DevelopingSuite(args)
    e = developingSuite.train_and_eval()
    developingSuite.writer.close()
