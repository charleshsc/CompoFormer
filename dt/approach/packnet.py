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
                iteration_split=3, prune_instructions=0.7,
                **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)
        
        self.prunable_types = (nn.Linear, nn.LayerNorm, Conv1D)
        self.iteration_split = iteration_split
        self.prune_instructions = prune_instructions

        # In all cases, we only keep importance weights for the model, but not for the heads.
        # Store current parameters as the initial parameters before first task starts
        self.masks = []
    
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument('--iteration_split', default=3, type=int, required=False,
                            help='first train percentage')
        parser.add_argument('--prune_instructions', default=0.7, type=float, required=False,
                            help='prune ratio')

        return parser.parse_known_args(args)
    
    def set_optimizer(self):
        """Returns the optimizer"""
        # params = self.model.parameters()
        embed_params = [param for name, param in self.model.main_named_parameters() if param.requires_grad]
        params = embed_params + list(self.model.heads[-1].parameters())

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

        self.model.train()
        for i in range(num_steps):
            if cur_i <= self.iteration_split:
                train_loss = self.train_iteration(env_name, iters, finetune=False)
            else:
                train_loss = self.train_iteration(env_name, iters, finetune=True)
            
            train_losses.append(train_loss)

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if cur_i == self.iteration_split:
            if iters == self.n_task - 1:
                self.mask_remaining_params()
            else:
                self.prune(self.prune_instructions)

        return logs
    
    def train_iteration(self, index, iters, finetune):
        self.model.train()

        loss = self.criterion(index, iters)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        if finetune:
            self.fine_tune_mask(iters)
        else:
            self.training_mask()

        self.optimizer.step()

        return loss.detach().cpu().item()
    
    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""
        
        # self.fix_biases() # Fix biases after first task
        self.fix_batch_norm() # Fix batch norm mean, var, and params
    
    def eval_iteration_metaworld(
            self, 
            eval_episodes, 
            env_name_list, 
            info,
            variant, 
            env_list, 
            iter_num=0,  
            group='test',
        ):
        
        self.logger.log('=' * 80)
        self.logger.log('evaluate at tasks: ')
        for i in env_name_list:
            self.logger.log(i)
        logs = dict()
        self.logger.log('start evaluating...')
        
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            
            # need to sample eval_fn and prompt together 
            self.logger.log(f'Evaluate at task: {env_name}')
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            
            if len(self.masks) > 0:
                model = copy.deepcopy(self.model)
                mask_id = min(env_id, len(self.masks)-1)
                self.apply_eval_mask(model, mask_id)
                for eval_fn in self.eval_fns:
                    outputs = eval_fn(model, info=info)
                    for k, v in outputs.items():
                        logs[f'{group}-evaluation/{k}'] = v
            else:
                for eval_fn in self.eval_fns:
                    outputs = eval_fn(self.model, info=info)
                    for k, v in outputs.items():
                        logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        total_return_mean = {}
        total_success_mean = {}
        self.logger.record_tabular('Iteration', iter_num)
        for k, v in logs.items():
            # self.logger.record_tabular(k, float(v))
            if 'return_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + k.split('/')[1].split('_')[1]
                if env not in total_return_mean.keys():
                    total_return_mean[env] = float(v)
                elif total_return_mean[env] < float(v):
                    total_return_mean[env] = float(v)
            if 'success_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + k.split('/')[1].split('_')[1]
                if env not in total_success_mean.keys():
                    total_success_mean[env] = float(v)
                elif total_success_mean[env] < float(v):
                    total_success_mean[env] = float(v)        
        
        total_mean = []
        total_success = []
        for k, v in total_return_mean.items():
            self.logger.record_tabular(f'{group}-{k}-Return', float(v))
            self.logger.record_tabular(f'{group}-{k}-Success', float(total_success_mean[k]))
            total_mean.append(v)
            total_success.append(total_success_mean[k])
        self.logger.dump_tabular()

        logs['total_success'] = total_success

        return logs

    
    def prune(self, prune_quantile):
        all_prunable = torch.tensor([]).to(self.device)
        for mod_name, mod in self.model.main_named_modules():
            if isinstance(mod, self.prunable_types):
                for name, param_layer in mod.named_parameters():
                    # if 'bias' not in name:
                    # get fixed weights for this layer
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)

                    for task in self.masks:
                        if mod_name+name in task:
                            prev_mask |= task[mod_name+name]
                    
                    p = param_layer.masked_select(~prev_mask)

                    if p is not None:
                        all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        
        cutoff = torch.quantile(torch.abs(all_prunable), q=prune_quantile)

        mask = {} # create mask for this task
        with torch.no_grad():
            for mod_name, mod in self.model.main_named_modules():
                if isinstance(mod, self.prunable_types):
                    for name, param_layer in mod.named_parameters():
                        # if 'bias' not in name:
                        # get weight mask for this layer
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)  # p

                        for task in self.masks:
                            if mod_name + name in task:
                                prev_mask |= task[mod_name+name]

                        curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                        curr_mask = torch.logical_and(curr_mask, ~prev_mask)  # (q & ~p)

                        # Zero non masked weights
                        param_layer *= (curr_mask | prev_mask)

                        mask[mod_name+name] = curr_mask
        
        self.masks.append(mask)
    
    def fine_tune_mask(self, current_task_id):
        """
        Zero the gradient of pruned weights this task as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        """
        assert len(self.masks) > current_task_id

        mask_idx = 0
        for mod_name, mod in self.model.main_named_modules():
            if isinstance(mod, self.prunable_types):
                for name, param_layer in mod.named_parameters():
                    if param_layer.grad is None:
                        self.logger.log(f'grad none is {mod_name}.{name}')
                    # if 'bias' not in name:
                    param_layer.grad *= self.masks[current_task_id][mod_name+name]
                    mask_idx += 1

    def training_mask(self):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        """
        if len(self.masks) == 0:
            return

        for mod_name, mod in self.model.main_named_modules():
            if isinstance(mod, self.prunable_types):
                for name, param_layer in mod.named_parameters():
                    # if 'bias' not in name:
                    # get mask of weights from previous tasks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)

                    for task in self.masks:
                        prev_mask |= task[mod_name + name]

                    # zero grad of previous fixed weights
                    param_layer.grad *= ~prev_mask

    def fix_biases(self):
        """
        Fix the gradient of prunable bias parameters
        """
        for mod in self.model.main_named_modules():
            if isinstance(mod, self.prunable_types):
                for name, param_layer in mod.named_parameters():
                    # if 'bias' in name:
                    param_layer.requires_grad = False

    def fix_batch_norm(self):
        """
        Fix batch norm gain, bias, running mean and variance
        """
        for mod in self.model.main_named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False
    
    def apply_eval_mask(self, model, task_idx):
        """
        Revert to network state for a specific task
        :param model: the model to apply the eval mask to
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        """

        assert len(self.masks) > task_idx

        with torch.no_grad():
            for mod_name, mod in model.main_named_modules():
                if isinstance(mod, self.prunable_types):
                    for name, param_layer in mod.named_parameters():
                        # if 'bias' not in name:

                        # get indices of all weights from previous masks
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                        for i in range(0, task_idx + 1):
                            prev_mask |= self.masks[i][mod_name + name]

                        # zero out all weights that are not in the mask for this task
                        param_layer *= prev_mask
    

    def mask_remaining_params(self):
        """
        Create mask for remaining parameters
        """
        mask = {}
        for mod_name, mod in self.model.main_named_modules():
            if isinstance(mod, self.prunable_types):
                for name, param_layer in mod.named_parameters():        
                    # if 'bias' not in name:

                    # Get mask of weights from previous tasks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                    for task in self.masks:
                        prev_mask |= task[mod_name + name]

                    # Create mask of remaining parameters
                    layer_mask = ~prev_mask
                    mask[mod_name + name] = layer_mask

        self.masks.append(mask)