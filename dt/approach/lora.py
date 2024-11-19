import torch
from torch import nn
import numpy as np
from argparse import ArgumentParser
from transformers.modeling_utils import Conv1D
from .incremental_learning import Inc_Learning_Appr
import time
import copy

class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)
        
    
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()

        return parser.parse_known_args(args)
    
    def set_optimizer(self, task_id=0):
        """Returns the optimizer"""
        if task_id == 0:
            embed_params = [param for name, param in self.model.main_named_parameters() if param.requires_grad]
            params = embed_params + list(self.model.heads[-1].parameters())
        else:
            params = self.model.transformer.get_lora_parameters(task_id) + list(self.model.heads[-1].parameters())

        optimizer = torch.optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        self.optimizer = optimizer
    
    def train_loop(self, num_steps, env_name=None, iters=0, cur_i=0):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.set_optimizer(iters)
        self.model.train()
        for i in range(num_steps):
            train_loss = self.train_iteration(env_name, iters)
            train_losses.append(train_loss)

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        return logs
    
    def train_iteration(self, index, iters):
        self.model.train()

        loss = self.criterion(index, iters)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()
    
    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""
        
        self.model.add_lora()
        self.model.to(device=self.device)
    
    