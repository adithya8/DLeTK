import torch
import torch.nn as nn
from typing import Dict, List, Union

from .block import *
        
class PytorchModelBuilder2(nn.Module):
    def __init__(self, model_architecture: Dict):
        super().__init__()
        self.model_architecture = model_architecture
        self.parse_model_architecture()
    
    def forward(self, **kwargs):
        """
            Forward Pass of the model
        """
        forward_output_features = self.model_architecture["output_names"]
        for layer_name in self.model_architecture['architecture'].keys():
            layer = self.get_submodule(layer_name)
            layer_input_names = self.model_architecture['architecture'][layer_name]["input_names"]
            layer_output_names = self.model_architecture['architecture'][layer_name]["output_names"]
            layer_inputs = [kwargs[input_name] if input_name in kwargs else self.__dict__[input_name] for input_name in layer_input_names]
            if len(layer_output_names) == 1: 
                self.__dict__[layer_output_names[0]] = layer(*layer_inputs)
            else:
                layer_outputs = layer(*layer_inputs)
                self.__dict__.update({output_name: output for output_name, output in zip(layer_output_names, layer_outputs)})
        
        assert all([(output_name in self.__dict__ or output_name in kwargs) for output_name in forward_output_features]), \
            'Output features %s not found in the forward pass' % ','.join([output_name for output_name in forward_output_features if (output_name not in self.__dict__ and output_name not in kwargs)])
                                        
        return {output_name: self.__dict__[output_name] if output_name in self.__dict__ else kwargs[output_name] for output_name in forward_output_features}

    def parse_model_architecture(self) -> None:
        """
            Parses the model architecture and adds the layers to the model
        """
        for layer_name, layer in self.model_architecture['architecture'].items():
            self.add_module(layer_name, eval(f"{layer['module']}(**{layer['module_args']})"))
            
    def save_model(self, model_dir_path: str) -> None:
        """
            Saves the model to the model_path
        """
        raise NotImplementedError("Saving model is not implemented for Pytorch Model Builder")
    
    def load_model(self, model_dir_path: str) -> None:
        """
            Loads the model from the model_path
        """
        raise NotImplementedError("Loading model is not implemented for Pytorch Model Builder")
    
    @staticmethod
    def from_pretrained(model_dir_path: str):
        """
            Loads the model from the model_path
        """
        raise NotImplementedError("Loading model is not implemented for Pytorch Model Builder")
    