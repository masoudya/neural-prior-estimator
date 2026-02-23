import torch
import torch.nn as nn
import torch.nn.init as init


def weights_init(m):
    """
    - Conv2d: Kaiming He
    - Linear, auxilaries: Xavier
    - BatchNorm: weight=1, bias=0
    """
    std = 0.001
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
            
    elif isinstance(m, nn.ModuleList):
        for i in m:
            init.normal_(i.weight, std = std)
            if i.bias is not None:
                init.zeros_(i.bias)
        if m:
            print(f'PEM weights initialized by normal dis. with std = {std}')
