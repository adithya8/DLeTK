examples = {}

examples['simple_1lyr'] = {
    "framework": "pytorch",
    "network_name": "test_network", 
    "input_names": ["input_features"],
    "output_names": ["output_0"],
    "architecture": {
        "layer_0": {
            "input_names": ["input_features"],
            "module": "nn.Linear",
            "module_args": {
                "in_features": 768, 
                "out_features": 3
                },
            "output_names": ["output_0"]
            },
        },
}

examples['simple_2lyr'] = {
    "framework": "pytorch",
    "network_name": "test_network", 
    "input_names": ["input_features"],
    "output_names": set(["output_1"]),
    "architecture": {
        "layer_0": {
            "input_names": ["input_features"],
            "module": "nn.Linear",
            "module_args": {
                "in_features": 768, 
                "out_features": 3
                },
            "output_names": ["output_0"]
            },
        "layer_1": {
            "input_names": ["output_0"],
            "module": "nn.Linear",
            "module_args": {
                "in_features": 3,
                "out_features": 1
                },
            "output_names": ["output_1"],
            },
    },
}

examples['simple_1lyr_with_activation'] = {
    "framework": "pytorch",
    "network_name": "test_network",
    "input_names": ["input_features"],
    "output_names": ["output_0", "output_1"],
    "architecture": {
        "layer_0": {
            "input_names": ["input_features"],
            "module": "nn.Linear",
            "module_args": {
                "in_features": 768,
                "out_features": 3
            },
            "output_names": ["output_0"]
        },
        "layer_1": {
            "input_names": ["output_0"],
            "module": "nn.ReLU",
            "module_args": {},
            "output_names": ["output_1"],
        },
    }
}

examples['simple_1lyr_with_activation_async_input'] = {
    "framework": "pytorch",
    "network_name": "test_network",
    "input_names": ["input_features", "async_input"],
    "output_names": ["output_0", "output_1", "output_2"],
    "architecture": {
        "layer_0": {
            "input_names": ["input_features"],
            "module": "nn.Linear",
            "module_args": {
                "in_features": 768,
                "out_features": 3
            },
            "output_names": ["output_0"]
        },
        "layer_1": {
            "input_names": ["async_input"],
            "module": "nn.ReLU",
            "module_args": {},
            "output_names": ["output_1"],
        },
        "layer_2": {
            "input_names": ["output_0", "output_1"],
            "module": "Cat",
            "module_args": {
                "dim": 1
            },
            "output_names": ["output_2"],
        },
    },
}

from main import parse_model_architecture
import torch

sample = torch.randn(5, 768)
async_sample = torch.randn(5, 3)
input_features_dict = {"input_features": sample}

# model = parse_model_architecture(examples['simple_1lyr'])
# print (model)
# print (model(input_features_dict))
# print ('---------------------------------------------------------------------------')
# model = parse_model_architecture(examples['simple_2lyr'])
# print (model)
# print (model(input_features_dict))
# print ('---------------------------------------------------------------------------')
# model = parse_model_architecture(examples['simple_1lyr_with_activation'])
# print (model)
# print (model(input_features_dict))
# print ('---------------------------------------------------------------------------')
model = parse_model_architecture(examples['simple_1lyr_with_activation_async_input'])
print (model)
print (model({"input_features": sample, "async_input": async_sample}))

