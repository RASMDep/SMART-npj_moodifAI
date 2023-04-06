import sys
import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import random

'''
Self-learning with Multi-Prototypes to train a network on a real noisy dataset without extra supervision.
ref:
Han, Jiangfan, Ping Luo, and Xiaogang Wang. "Deep self-learning from noisy labels."
Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
'''

class selfLC():

    def __init__(self, args, data_train ):

        self.args = args
        self.data_train = data_train
        self.alpha_label = args.alpha_label
        self.percent_samples = args.percent_samples
        self.num_prototypes = args.num_prototypes
        self.selection_threshold = args.selection_threshold
        self.threshold_correction = args.threshold_correction
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def LC_prototypes(self, model):

        model.eval()
        # get all samples from each class label
        self.num_classes =  5# self.data_train.num_classes
        self.classes = self.data_train.classes
        self.labels = self.data_train.labels
        merged = list(itertools.chain(*self.labels))
        num_labels_class = [merged.count(l) for l in self.classes]

        classes_list = [[
                idx for idx, c in enumerate(self.labels) if unique_class in c
            ] for unique_class in self.classes]

        self.classes_present = [
                idx for idx, c in enumerate(classes_list) if len(c) > 0
            ]

        # get random samples from each class
        class_inds = [
            random.sample(list(classes_list[i]),
                              int(self.percent_samples * num_labels_class[i])) if int(self.percent_samples * num_labels_class[i]) > 1
                              else random.sample(list(classes_list[i]), int(num_labels_class[i]))
                for i in range(self.num_classes) if len(classes_list[i]) > 0
            ]

        self.dataloaders = [
                DataLoader(dataset=Subset(self.data_train, inds),
                           batch_size=10,
                           shuffle=True,
                           drop_last=False) for inds in class_inds
            ]

        self.num_l_c = [i for i in num_labels_class if i != 0]

        # function to call the hook signature and store the output
        self.activation = {}
        # get hook for layer and features, layer where we want the hook for feature extraction (before FC)
        model.lastpool.register_forward_hook(  #
        self.get_activation('lastpool'))  #lastpool
        # get random samples
        self.get_random_samples(model)
        # prototype selection
        self.prototype_selection()

        return


    def get_activation(self, name):
        def hook(model,input, output):
            self.activation[name] = output.detach()
        return hook


    def get_random_samples(self, model):
        # get features for the random samples of each class
        self.features_classes = [] #defaultdict(list)
        self.features_class_labels = []

        for en2, c in enumerate(self.dataloaders):
            self.features_class = []
            for sample2 in c:
                x2 = sample2["x"]
                x2= x2.to(self.device)
                model(sample2)
                features_y = self.activation['lastpool'].data.cpu().numpy()  #lastpool
                # get rid of one dimension
                fatures_y = features_y[:,:,0]
                self.features_class.append(fatures_y)
                self.features_class_labels.append(sample2["labels"])

            self.features_classes.append(np.vstack(self.features_class))

        return

    def prototype_selection(self):

        self.prototypes_classes = []
        for X in self.features_classes:
            n_features = X.shape[1]
            # compute similarities and densities
            #S = [[cosine_similarity_numba(f1,f2) for f1 in X] for f2 in X]
            #S = np.vstack(S)
            S = cosine_similarity(X)
            s_c = np.quantile(S, 0.75)
            sign_matrix = np.sign(S - s_c)
            densities = np.sum(sign_matrix, axis=0)

            # sort densities in descending order
            idx = (-densities).argsort()

            # select prototypes
            prototypes = np.zeros((self.num_prototypes, n_features), dtype=np.float32)
            prototypes[0] = X[idx[0]]
            prototypes_idx = [idx[0]]
            prototypes_found = 1

            # selection is slightly different than in the paper, similarities are only considered to other samples that were selected as prototypes
            # In my (Julians) opinion this makes more sense
            for e, i in enumerate(idx[1:]):
                if prototypes_found >= self.num_prototypes:
                    break
                valid_values = [S[i, j] for j in prototypes_idx]
                # valid_values = [S[i, j] for j in idx[:(e+1)]]
                sim_measure = np.min(valid_values)
                if sim_measure < self.selection_threshold:
                    prototypes[prototypes_found] = X[i]
                    prototypes_idx.append(i)
                    prototypes_found += 1

            self.prototypes_classes.append(prototypes)

        self.prototypes_classes = [torch.from_numpy(p).to(self.device) for p in self.prototypes_classes]

        return

    def correctLabels(self,model,y):

        # get hook for layer and features, layer where we want the hook for feature extraction (before FC)
        model.lastpool.register_forward_hook(self.get_activation('lastpool')) #lastpool
        features_x = self.activation['lastpool'].data#lastpool
        features_sample = torch.squeeze(features_x)

        self.features_sample=features_sample

        feat_norm = self.features_sample / torch.norm(self.features_sample, dim=1)[:, None]
        prot_norm = [ prot / torch.norm(prot,dim=1)[:, None]for prot in self.prototypes_classes]
        self.similarities = [ torch.matmul( feat_norm, prototypes.T) for prototypes in prot_norm]

        similarity_scores  = [ torch.mul(torch.sum(sim,dim=1), 1/self.num_prototypes) for sim in self.similarities  ]

        self.similarity_scores = torch.vstack(similarity_scores)

        # correct labels
        if self.threshold_correction:

            array = np.zeros((self.similarity_scores.shape[1],self.num_classes), dtype='float32')
            for b in range(0,self.args.batch_size):
                max = torch.max(self.similarity_scores[:,b])
                keep_idxs = self.similarity_scores[:,b]>=(0.9)
                if keep_idxs[0] and keep_idxs[1]:
                    both = torch.tensor([self.similarity_scores[0,b],self.similarity_scores[1,b]])
                    id_remove = torch.argmin(both)
                    keep_idxs[id_remove] = False
                    
                array[b][np.array(self.classes_present)[np.where(keep_idxs.cpu())[0]]] = 1
        else:
            y_corrected = torch.argmax(self.similarity_scores,dim=0).cpu().numpy()
            # transform it to a tensor representation
            array = np.zeros((len(features_sample), self.num_classes), dtype='float32')
            for i, f in enumerate(y_corrected):
                array[i][f] = 1

        self.y_c = torch.from_numpy(array)  #corrected label


        return self.y_c

