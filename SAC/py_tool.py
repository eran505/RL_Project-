from torch import nn
import torch
import math
''''
    'uniform': nn.init.uniform_,
    'normal': nn.init.normal_,
    'eye': nn.init.eye_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'he': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_
'''


# Initialize Policy weights
def weights_init_(m, name="xavier"):
    if name == "xavier":
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    elif name == "xavier_uniform":
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    elif name == "kaiming_normal":
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    elif name == "kaiming_uniform":
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)