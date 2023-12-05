# Import libraries
import sys, argparse, os
import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.utils import *
from data import get_data
from models import get_model
from models import *


if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm, trange

parser = argparse.ArgumentParser()

### general parameters ################################################
parser.add_argument( '--mode', default="test_train",
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
                    type=str,
                    help="root folder")
parser.add_argument('--tag', default="__", type=str)
parser.add_argument("--val-every-n-epochs",
                    type=int,
                    default=1,
                    help="interval of training epochs to run the validation")
parser.add_argument("--experiment-folder",
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
                    help="subject to leave out for test")
parser.add_argument("--n-kfold", type=int, default=5, help="How many folds?")

### data parameters ################################################
parser.add_argument("--target",
                    type=str,
                    choices=["valence_class", "arousal_class", "depression","kss_class"],
                    help="classification target")
parser.add_argument("--data",
                    type=str,
                    choices=["patch_data"],
                    help="dataset selection")
parser.add_argument("--data-file",
                    type=str,
                    help="dataset file")
parser.add_argument("--data-path",
                    type=str,
                    help="location of the data")
parser.add_argument("--resamp-freq",
                    type=int,
                    default=125,
                    help="resample the data to this frequency")


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
parser.add_argument("--batch-size",
                    type=int,
                    default=600,
                    help='Batch size for train, validation and test sets [default=%(default)s].')
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
parser.add_argument( "--weight-decay",
                    type=float,
                    default=0.,
                    help='Add L2 regularization to the loss function [default=%(default)s].')

### model parameters ################################################
parser.add_argument( "--model",
                    type=str,
                    default='LeNet',
                    choices=['Conv1dNet', 'AttConv1dNet'],
                    help='Model to use [default=%(default)s].')
parser.add_argument("--n-channels",
                    type=int,
                    default=12,
                    help="number of input channels/timeseries")
parser.add_argument("--num-class",
                    type=int,
                    help="number of output classes")

parser.add_argument("--n-covariates",
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



class DevelopingSuite(object):
    def __init__(self, args):

        self.args = args
        torch.manual_seed(0)

        self.data_train, self.data_val, self.data_test = get_data(args)

        # Define torch dataloader object
        self.train_dataloader = DataLoader(self.data_train, batch_size=args.batch_size,num_workers=args.num_workers, 
                                           drop_last=True,shuffle=True)
        self.val_dataloader = DataLoader(self.data_train, args.batch_size,num_workers=args.num_workers)

        #args.n_covariates = 0 

        # Use GPU if available
        print('GPU devices available:', torch.cuda.device_count())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        accumulated_loss, accumulated_func_eval = 0, 0

        with tqdm(self.train_dataloader, leave=False) as inner_tnr:
            for en, sample in enumerate(inner_tnr):

                y = sample['label'].to(self.device)#.unsqueeze(1)
                x = sample['x'].to(self.device)
                cov = sample['covariates'].to(self.device)
                self.optimizer.zero_grad()
                
                pred_y = self.model(x,cov)
                loss = self.model.loss(pred_y,y) 

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

        dataloader = self.val_dataloader
        self.model.eval()
        for sample in dataloader:
            with torch.no_grad():
                x1 = sample["x"].to(self.device)
                cov = sample["covariates"].to(self.device)
                y = sample["label"].to(self.device)#.unsqueeze(1)

                batch_size = x1.size(0)
                pred_y = self.model(x1, cov)
                # Add metrics of current batch
                total_dataset_size += batch_size
                total_loss += (self.model.loss(pred_y,y)  
                                )* batch_size 

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

        outputs_all= torch.zeros(0).to(self.device)  # <-- changed so it is automatic
        targets_all = torch.zeros(0).to(self.device)  # <-- changed so it is automatic
        scores_all = torch.zeros(0).to(self.device)  # <-- changed so it is automatic
        gt_all = torch.zeros(0).to(self.device)  # <-- changed so it is automatic

        tot = 0

        self.args.mode = "test"
        # Define torch dataloader object
        self.test_dataloader = DataLoader(self.data_test,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers)


        with torch.no_grad():
            for sample in tqdm(self.test_dataloader):
                x_t = sample["x"].to(self.device)
                cov = sample["covariates"].to(self.device)
                y = torch.argmax(sample["label"],dim=1).to(self.device)#.unsqueeze(1)
                
                tot = tot + x_t.shape[0]
                y_pred = self.model(x_t, cov)
                #y_pred_binary = 0.5 * (torch.sign(torch.sigmoid(y_pred) - 0.5)
                #                       + 1).to(self.device)
                y_pred_binary = torch.argmax(y_pred,dim=1)
                outputs_all= torch.cat((outputs_all, y_pred_binary), 0)
                targets_all = torch.cat((targets_all, y,), 0)
                scores_all = torch.cat((scores_all, y_pred,), 0)
                gt_all = torch.cat((gt_all, sample["label"].to(self.device),), 0)

        return outputs_all,targets_all,gt_all,scores_all


if __name__ == '__main__':

    args = parser.parse_args()

    args.covariates = []

    developingSuite = DevelopingSuite(args)
    e = developingSuite.train_and_eval()
    developingSuite.writer.close()