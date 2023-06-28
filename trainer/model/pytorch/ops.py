from typing import List, Union
import torch
import torch.nn as nn

class PyTorchOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        raise NotImplementedError("PyTorchOp is an abstract class. Please implement the forward method.")
    
############################################
## Some Basic Operations for Pytorch Model
############################################

class Cat(PyTorchOp):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, *args):
        return torch.cat(args, dim=self.dim)

class Add(PyTorchOp):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, *args):
        return [torch.sum(arg, dim=self.dim) for arg in args]

class Prod(PyTorchOp):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        
    def forward(self, *args):
        return [torch.prod(arg, dim=self.dim) for arg in args]

class Matmul(PyTorchOp):
    def __init__(self, transpose_second_matrix:bool=False) -> None:
        super().__init__()
        self.transpose_second_matrix = transpose_second_matrix
    
    def forward(self, matrix1, matrix2):
        if self.transpose_second_matrix:
            return torch.matmul(matrix1, matrix2.T)
        return torch.matmul(matrix1, matrix2)

class TensorDot(PyTorchOp):
    def __init__(self, dims: Union[int, List[int]]) -> None:
        super().__init__()
        self.dims = dims
        
    def forward(self, tensor1, tensor2):
        return torch.tensordot(tensor1, tensor2, dims=self.dims)
    
class Permute(PyTorchOp):
    def __init__(self, *dims) -> None:
        super().__init__()
        self.dims = dims
        
    def forward(self, *args):
        return [arg.permute(*self.dims) for arg in args]

class Transpose(PyTorchOp):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, *args):
        return [arg.transpose(self.dim0, self.dim1) for arg in args]

############################################

class HFTransformers(PyTorchOp):
    # TODO: Move HFTransformers outside since it is not a fundamental Operation and besides it is a feature extractor
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.model = None
    
    def forward(self):
        raise NotImplementedError("HFTransformers is an abstract class. Please implement the forward method.")