import math

import torch
import numpy as np
from typing import List


def get_npcer_at_apcer(target_apcer, npcer, apcer, thresholds):
    apcer_zero_idx = np.where(apcer == 0)[0]
    if len(apcer_zero_idx) < 1:
        return 1, 1

    apcer_zero_idx = apcer_zero_idx[-1]
    npcer_val = npcer[apcer_zero_idx]
    thresh_val = thresholds[apcer_zero_idx]

    return npcer_val, thresh_val


def calculate_acer(tp, fp, tn, fn):
    apcer = fp / (tn * 1.0 + fp * 1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer, apcer, npcer


def calculate_acer_flip(tp, fp, tn, fn):
    """Calculate ACER, APCER, and NPCER.
    This function differs from the calculate_acer where
    the live/spoof samples are labeled 0/1 instead of 1/0.
    """
    apcer = fn / (tp * 1.0 + fn * 1.0)
    npcer = fp / (fp * 1.0 + tn * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer, apcer, npcer


def calculate_acc(TP, TN, labels):
    ACC = (TP + TN) / labels.shape[0]
    return ACC


def confusion_matrix(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def get_thresholds(grid_density=10000):
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds


def get_EER_states(probs, labels, grid_density=10000):
    thresholds = get_thresholds(grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = confusion_matrix(probs, labels, thr)
        if FN + TP == 0:
            FRR = TPR = 1.0  # pylint: disable=unused-variable
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)  # pylint: disable=unused-variable
        elif FP + TN == 0:
            TNR = FAR = 1.0  # pylint: disable=unused-variable
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)  # pylint: disable=unused-variable
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)  # pylint: disable=unused-variable
            TPR = TP / float(TP + FN)  # pylint: disable=unused-variable
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list


def calculate_hter(tn, fn, fp, tp):
    if fn + tp == 0:
        FRR = 1.0
        FAR = fp / float(fp + tn)
    elif fp + tn == 0:
        FAR = 1.0
        FRR = fn / float(fn + tp)
    else:
        FAR = fp / float(fp + tn)
        FRR = fn / float(fn + tp)
    HTER = (FAR + FRR) / 2.0
    return HTER


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def topk_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
