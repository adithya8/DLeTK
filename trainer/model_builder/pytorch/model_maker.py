import torch
import torch.nn as nn
from typing import Dict, List, Union

from .block import *

class PytorchModelBuilder(nn.Module):
    def __init__(self, model_architecture: Dict):
        super().__init__()
        self.model_architecture = model_architecture
        # self.model = PytorchModelClass(name=self.model_architecture["network_name"]) 
        self.parse_model_architecture()
    
    def forward(self, input_features_dict: Dict[str, torch.Tensor]):
        """Forward pass of the model"""
        
        forward_output_features = set(self.model_architecture["output_names"])
        output_features = {}
        # Preparing the input features for the forward pass
        for input_feature_name in self.model_architecture["input_names"]:
            input_features_op = f"""{input_feature_name} = input_features_dict["{input_feature_name}"]"""
            exec(input_features_op)
            if input_feature_name in forward_output_features:
                output_features[input_feature_name] = input_features_dict[input_feature_name]
        
        # Forward pass
        for layer_name in self.model_architecture["architecture"].keys():
            layer = self.get_submodule(layer_name)
            layer_input_names = self.model_architecture['architecture'][layer_name]["input_names"]
            layer_output_names = self.model_architecture['architecture'][layer_name]["output_names"]
            layer_op = f"""{','.join(layer_output_names)} = layer({','.join(layer_input_names)});"""
            exec(layer_op)
            
            # Storing the output features in return dictionary
            for output in layer_output_names:
                if output in forward_output_features:
                    output_features[output] = eval(output)
                
        return output_features
        
    def parse_model_architecture(self) -> None:
        """Parses the model architecture and adds the layers to the model
        """
        for layer_name, layer in self.model_architecture["architecture"].items():
            self.add_module(layer_name, eval(f"{layer['module']}(**{layer['module_args']})"))
        