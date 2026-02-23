import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustment(nn.Module):

    def __init__(self, cls_num_list):
        super(LogitAdjustment, self).__init__()
        cls_prior = cls_num_list / torch.sum(cls_num_list)
        self.log_prior = torch.log(cls_prior + 1e-12).unsqueeze(0)
    def forward(self, logits, labels):
        
        return F.cross_entropy(logits + self.log_prior, labels) 
