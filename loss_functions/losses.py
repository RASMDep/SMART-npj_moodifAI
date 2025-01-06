'''
    Asymmetric Loss For Multi-Label Classification
        
    ref:
    Yun, Sangdoo, et al. "Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels." 
    arXiv preprint arXiv:2101.05022 (2021).

    https://github.com/Alibaba-MIIL/ASL
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([0.33, 0.33, 0.33])
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        
        #target = target.view(-1,1).long()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target.long())
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1).long())
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

        
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=1, gamma_pos=3, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
                #torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
                #torch._C.set_grad_enabled(True)

            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLoss2(torch.nn.Module):
    def __init__(self, gamma_pos=2, gamma_neg=6, clip=0.1, eps=1e-8):
        """
        ASL Loss
        Parameters:
        - gamma_pos: Focusing parameter for positive samples.
        - gamma_neg: Focusing parameter for negative samples.
        - clip: Clip value for probabilities to stabilize training.
        - eps: Small value to avoid numerical instability.
        """
        super(AsymmetricLoss2, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        Forward pass for ASL Loss.
        Parameters:
        - logits: Predicted logits (batch_size, num_classes).
        - targets: Ground truth (batch_size, num_classes), one-hot encoded or binary.
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Optionally clip probabilities for numerical stability
        if self.clip is not None:
            probs = probs.clamp(min=self.clip, max=1 - self.clip)

        # Compute positive and negative probabilities
        pos_probs = probs
        neg_probs = 1 - probs

        # Positive and negative losses
        pos_loss = targets * torch.log(pos_probs + self.eps) * (1 - pos_probs) ** self.gamma_pos
        neg_loss = (1 - targets) * torch.log(neg_probs + self.eps) * (1 - neg_probs) ** self.gamma_neg

        # Total loss
        loss = -torch.mean(pos_loss + neg_loss)
        return loss

class AsymmetricLossMultiClass(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=3, clip=0.05, eps=1e-8):
        super(AsymmetricLossMultiClass, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits of shape (N, C)
        y: targets of shape (N,) where each value is in range [0, C-1]
        """

        # Softmax probabilities
        x_softmax = torch.softmax(x, dim=1).long()

        # Gather probabilities for the true class
        true_probs = x_softmax[range(x.size(0)), y.long()]

        # Compute probabilities for the other classes
        other_probs = x_softmax.clone()
        other_probs[range(x.size(0)), y] = 0  # Exclude true class probabilities

        # Asymmetric Clipping for negative class probabilities
        if self.clip is not None and self.clip > 0:
            other_probs = (other_probs + self.clip).clamp(max=1)

        # Loss for the positive (true) class
        pos_loss = torch.log(true_probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_loss *= torch.pow(1 - true_probs, self.gamma_pos)

        # Loss for the negative (other) classes
        neg_loss = torch.sum(torch.log((1 - other_probs).clamp(min=self.eps)), dim=1)
        if self.gamma_neg > 0:
            neg_loss *= torch.pow(other_probs, self.gamma_neg).sum(dim=1)

        # Combine positive and negative losses
        loss = -pos_loss - neg_loss

        return loss.mean()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss