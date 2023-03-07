import torch
import torch.nn as nn

class PytorchBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        raise NotImplementedError("TorchBlock is an abstract class. Please implement the forward method.")
    
############################################
## Some Basic Operations for Pytorch Model
############################################

class Cat(PytorchBlock):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, *args):
        return torch.cat(args, dim=self.dim)

class Add(PytorchBlock):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, *args):
        return torch.sum(args, dim=self.dim)

class Prod(PytorchBlock):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, *args):
        return torch.prod(args, dim=self.dim)

############################################