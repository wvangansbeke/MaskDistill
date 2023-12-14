#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyOHEM(nn.Module):
    """
        Cross-entropy loss with online hard example mining.
        
        Arguments:
            threshold: Float between 0 and 1. that determines the number of pixels that we select.
    """
    def __init__(self, threshold):
        super(CrossEntropyOHEM, self).__init__()
        self.loss = nn.CrossEntropyLoss(None, ignore_index = 255, reduction='none')
        self.threshold = threshold

    def forward(self, prediction, ground_truth):
        losses = self.loss(prediction, ground_truth.long())
        mask = (ground_truth != 255)
        losses_valid = torch.masked_select(losses, mask)
        num_hard = min(int(self.threshold*losses.numel()), losses_valid.numel())
        losses_ohem = torch.topk(losses_valid, num_hard)[0]
        return torch.mean(losses_ohem)
