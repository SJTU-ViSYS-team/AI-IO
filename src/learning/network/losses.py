import numpy as np
import torch
from torch import nn

"""
Log Likelihood loss, with logstdariance (only support diag logstd)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_logstd meaning:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""

MIN_LOG_STD = np.log(1e-3)
def loss_distribution_diag(pred, pred_logstd, targ):

    pred_logstd = torch.maximum(pred_logstd, MIN_LOG_STD * torch.ones_like(pred_logstd))
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
    return loss

"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""
def get_loss(pred, pred_logstd, targ, epoch, learn_configs):
    if learn_configs["switch_iter"] is not None:
        switch_epoch = learn_configs["switch_iter"]
        if epoch < switch_epoch:
            loss_type = learn_configs["loss_type"]
            if loss_type == "huber":
                loss = nn.functional.huber_loss(pred, targ, reduction='none', delta=learn_configs["huber_vel_loss_delta"])
            elif loss_type == "mse":
                loss = nn.functional.mse_loss(pred, targ, reduction='none')
            else:
                AssertionError("Unknown loss function!")
        else:
            loss = loss_distribution_diag(pred, pred_logstd, targ)
    else:
        loss_type = learn_configs["loss_type"]
        if loss_type == "huber":
            loss = nn.functional.huber_loss(pred, targ, reduction='none', delta=learn_configs["huber_vel_loss_delta"])
        elif loss_type == "mse":
            loss = nn.functional.mse_loss(pred, targ, reduction='none')
        else:
            AssertionError("Unknown loss function!")
    
    return loss