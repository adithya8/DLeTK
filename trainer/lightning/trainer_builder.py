trainer_dict = {
    "framework": "lightning",
    "trainer_name": "test_trainer",
    "trainer_args": {
        "max_epochs": 10,
        "gpus": 1,
        "distributed_backend": "dp",
        "precision": 16,
        "gradient_clip_val": 0.5,
        "logger": False,
        "checkpoint_callback": False,
        "callbacks": [],
        "num_sanity_val_steps": 0
    },
    "loss_fn": {
        "module": "torch.nn.functional.mse_loss",
        "module_args": {
            "reduction": "mean"
        },
    },
    "optimizer": {
        "optimizer_name": "torch.optim.AdamW",
        "optimizer_args": {
            "lr": 0.001, 
            "betas": (0.9, 0.999),
            "weight_decay": 0.01,
        }
    },
}

from typing import Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

def get_optimizer(optimizer_dict: Dict):
    # optimizer = getattr(torch.optim, optimizer_dict["optimizer_name"])
    optimizer = exec(optimizer_dict["optimizer_name"])
    optimizer_args = optimizer_dict["optimizer_args"]
    return optimizer(**optimizer_args)

def compute_loss(input_dict:Dict, loss_fn_dict: Dict):
    """
     Returns the loss function by setting the default arguments as given in the loss_fn_dict
    """
    def get_loss_fn(module_name:str):
        loss_fn = exec(module_name)
        return loss_fn
    return get_loss_fn(loss_fn_dict["module"])(**{**input_dict, **loss_fn_dict["module_args"]})
    
    

class LightningTrainer(pl.LightningModule):
    """
        
    """
    def __init__(self, model: nn.Module, trainer_dict: Dict):
        super().__init__()
        self.trainer_dict = trainer_dict
        self.model = model
    
    def forward(self, input_features_dict: Dict[str, torch.Tensor]):
        return self.model(input_features_dict)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = get_optimizer(self.trainer_dict["optimizer"])
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_outputs = self(batch)
        loss = compute_loss(batch_outputs, self.trainer_dict["loss_fn"])
        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_outputs = self(batch)
        loss = compute_loss(batch_outputs, self.trainer_dict["loss_fn"])
        return loss
    
    def testing_step(self, batch, batch_idx):
        batch_outputs = self(batch)
        loss = compute_loss(batch_outputs, self.trainer_dict["loss_fn"])
        return loss
    
    def train_dataloader(self):
        return 
    
    def val_dataloader(self):
        return 
    
    def test_dataloader(self):
        return 
    
    def training_epoch_end(self, outputs):
        pass
    
    def validation_epoch_end(self, outputs):
        pass
    
    def testing_epoch_end(self, outputs):
        pass
    
    def on_train_start(self):
        pass
    
    def on_train_end(self):
        pass
    
    def on_validation_start(self):
        pass
    
    def on_validation_end(self):
        pass
    
    def on_test_start(self):
        pass
    
    def on_test_end(self):
        pass
    
    def on_epoch_start(self):
        pass
    
    def on_epoch_end(self):
        pass
    
    def on_batch_start(self):
        pass
    
    def on_batch_end(self):
        pass
    
    def on_keyboard_interrupt(self):
        pass
    
    def on_save_checkpoint(self):
        pass
    
    def on_load_checkpoint(self):
        pass
    
    def on_before_zero_grad(self):
        pass
    
    def on_after_backward(self):
        pass
    
    def on_before_optimizer_step(self):
        pass
    
    def on_after_optimizer_step(self):
        pass
    
    def on_before_accelerator_backend_setup(self):
        pass
    
    def on_before_tpu_training(self):
        pass
    
    def on_tpu_training_end(self):
        pass
    
    def on_before_distributed_backend_setup(self):
        pass
    
    def on_before_dataloader(self):
        pass
    
    def on_after_dataloader(self):
        pass
    
    def on_before_accelerator_backend_teardown(self):
        pass
    
    def on_before_distributed_backend_teardown(self):
        pass
    
    def on_before_model_forward(self):
        pass
    
    def on_after_model_forward(self):
        pass
    
    def on_before_model_validation(self):
        pass
    
    def on_after_model_validation(self):
        pass
    
    def on_before_backward(self):
        pass
    
    def on_after_backward(self):
        pass
    
    def on_before_sanity_check(self):
        pass
    
    def on_after_sanity_check(self):
        pass
    
    def on_before_batch_transfer(self):
        pass
    
    def on_after_batch_transfer(self):
        pass
    
    
        