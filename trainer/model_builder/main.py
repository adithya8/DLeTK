# import torch
# import torch.nn as nn

from typing import Dict, List, Union

from pytorch import PytorchModelBuilder

def parse_model_architecture(model_architecture: Dict):
    
    if model_architecture["framework"] == "pytorch":
        return PytorchModelBuilder(model_architecture)
    else:
        raise NotImplementedError(f"Framework {model_architecture['framework']} is not implemented. Available frameworks are: pytorch")     