import os
import json
from typing import Dict, List, Union
import torch
import torch.nn as nn

from .ops import *
        
class PytorchModelBuilder(nn.Module):
    def __init__(self, model_architecture: Dict):
        """
            Initializes the model builder
            
            Parameters
            -----------
                model_architecture: Dict - Dictionary containing the model architecture
            
            Returns
            --------
                PytorchModelBuilder - Instance of the nn.Module
        """
        super().__init__()
        self.model_architecture = model_architecture
        self.parse_model_architecture()
    
    def forward(self, **kwargs):
        """
            Forward Pass of the model
            
            Parameters
            -----------
                kwargs: Dict[str, torch.Tensor] - Dictionary of input features
            
            Returns
            --------
                Dict[str, torch.Tensor] - Dictionary of output features
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
        if 'architecture' not in self.model_architecture:
            return 
        for layer_name, layer in self.model_architecture['architecture'].items():
            self.add_module(layer_name, eval(f"{layer['module']}(**{layer['module_args']})"))
            
    def save_model(self, model_dir_path: str) -> None:
        """
            Saves the model to the model_path
            
            Parameters
            -----------
                model_dir_path: str - Path to the model directory
        """
        if os.path.exists(model_dir_path):
            raise ValueError("Model directory already exists")
        
        os.makedirs(model_dir_path)
        torch.save(self.state_dict(), os.path.join(model_dir_path, "model.pt"))
        
        with open(os.path.join(model_dir_path, "model_architecture.json"), "w") as f:
            json.dump(self.model_architecture, f)
    
    def load_model(self, model_dir_path: str) -> None:
        """
            Loads the model from the model_path
        """
        if not os.path.exists(model_dir_path):
            raise ValueError("Model directory does not exists")
        
        try:
            with open(os.path.join(model_dir_path, "model_architecture.json"), "r") as f:
                self.model_architecture = json.load(f)
        except FileNotFoundError:
            raise ValueError("Model architecture file not found")
        
        self.parse_model_architecture()
        self.load_state_dict(torch.load(os.path.join(model_dir_path, "model.pt")), strict=False)
    
    @staticmethod
    def from_pretrained(model_dir_path: str) -> nn.Module:
        """
            Loads the model from the model_path
            
            Parameters
            -----------
                model_dir_path: str - Path to the model directory
            
            Returns
            --------
                nn.Module - Instance of the nn.Module
        """
        model = PytorchModelBuilder({})
        model.load_model(model_dir_path)
        return model
    