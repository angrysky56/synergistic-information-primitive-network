import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.module import Module as Module2


class Test1(nn.Module):
    pass


class Test2(Module):
    pass


class Test3(Module2):
    pass
