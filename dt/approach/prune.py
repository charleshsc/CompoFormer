import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from argparse import ArgumentParser
from transformers.modeling_utils import Conv1D
from .incremental_learning import Inc_Learning_Appr
import time
import copy

from sentence_transformers import SentenceTransformer

def merge_dictionaries(dicts):
    # 初始化一个新的字典来存储结果
    merged_dict = {}
    
    # 获取所有字典的key
    keys = set(key for d in dicts for key in d)
    
    # 对每个key进行合并
    for key in keys:
        merged_value = 0
        for d in dicts:
            if key in d:
                merged_value |= d[key]
        merged_dict[key] = merged_value
    
    return merged_dict

task_descrption = {
        "basketball-v2": "Dunk the basketball into the basket.",
        "bin-picking-v2": "Grasp the puck from one bin and place it into another bin.",
        "button-press-topdown-v2": "Press a button from the top.",
        "button-press-v2": "Press a button.",
        "button-press-wall-v2": "Bypass a wall and press a button.",
        "coffee-button-v2": "Push a button on the coffee machine.",
        "coffee-pull-v2": "Pull a mug from a coffee machine.",
        "coffee-push-v2": "Push a mug under a coffee machine.",
        "dial-turn-v2": "Rotate a dial 180 degrees.",
        "disassemble-v2": "pick a nut out of the a peg.",
        "door-close-v2": "Close a door with a revolving joint.",
        "door-lock-v2": "Lock the door by rotating the lock clockwise.",
        "door-open-v2": "Open a door with a revolving joint.",
        "door-unlock-v2": "Unlock the door by rotating the lock counter-clockwise.",
        "hand-insert-v2": "Insert the gripper into a hole.",
        "drawer-close-v2": "Push and close a drawer.",
        "drawer-open-v2": "Open a drawer.",
        "faucet-open-v2": "Rotate the faucet counter-clockwise.",
        "faucet-close-v2": "Rotate the faucet clockwise.",
        "handle-press-side-v2": "Press a handle down sideways.",
        "handle-press-v2": "Press a handle down.",
        "handle-pull-side-v2": "Pull a handle up sideways.",
        "handle-pull-v2": "Pull a handle up.",
        "lever-pull-v2": "Pull a lever down 90 degrees.",
        "peg-insert-side-v2": "Insert a peg sideways.",
        "pick-place-wall-v2": "Pick a puck, bypass a wall and place the puck.",
        "pick-out-of-hole-v2": "Pick up a puck from a hole.",
        "reach-v2": "reach a goal position.",
        "push-back-v2": "Pull a puck to a goal.",
        "push-v2": "Push the puck to a goal.",
        "pick-place-v2": "Pick and place a puck to a goal.",
        "plate-slide-v2": "Slide a plate into a cabinet.",
        "plate-slide-side-v2": "Slide a plate into a cabinet sideways.",
        "plate-slide-back-v2": "Get a plate from the cabinet.",
        "plate-slide-back-side-v2": "Get a plate from the cabinet sideways.",
        "soccer-v2": "Kick a soccer into the goal.",
        "push-wall-v2": "Bypass a wall and push a puck to a goal.",
        "shelf-place-v2": "pick and place a puck onto a shelf.",
        "sweep-into-v2": "Sweep a puck into a hole.",
        "sweep-v2": "Sweep a puck off the table.",
        "window-open-v2": "Push and open a window.",
        "window-close-v2": "Push and close a window.",
        "assembly-v2": "Pick up a nut and place it onto a peg.",
        "button-press-topdown-wall-v2": "Bypass a wall and press a button from the top.",
        "hammer-v2": "Hammer a screw on the wall.",
        "peg-unplug-side-v2": "Unplug a peg sideways.",
        "reach-wall-v2": "Bypass a wall and reach a goal.",
        "stick-push-v2": "Grasp a stick and push a box using the stick.",
        "stick-pull-v2": "Grasp a stick and pull a box with the stick.",
        "box-close-v2": "Grasp the cover and close the box with it."
    }

class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                iteration_split=3, prune_instructions=0.5, threshold=0.8,
                env_name_list=None, updateqk=False,
                **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)
        
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')
        self.task_descrption = task_descrption # output 384 dim

        self.prunable_types = (nn.Linear, nn.LayerNorm, Conv1D)
        self.iteration_split = iteration_split
        self.prune_instructions = prune_instructions
        self.threshold = threshold
        self.updateqk = updateqk

        self.env_name_list = env_name_list

        self.masks = []
        self.prev_mask = {}
        self.flag = True

    
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument('--iteration_split', default=3, type=int, required=False,
                            help='first train percentage')
        parser.add_argument('--prune_instructions', default=0.5, type=float, required=False,
                            help='prune ratio')
        parser.add_argument('--qkdim', default=256, type=int, required=False,)
        parser.add_argument('--threshold', default=0.8, type=float, required=False,)
        parser.add_argument('--proj_bias', default=True, type=bool, required=False,
                            help='Whether to activate bias in the linear transformations of the attention heads.')
        parser.add_argument('--updateqk', action='store_true', default=False)

        return parser.parse_known_args(args)
    
    def set_optimizer(self):
        """Returns the optimizer"""
        # params = self.model.parameters()
        embed_params = [param for name, param in self.model.main_named_parameters() if param.requires_grad]
        params = embed_params + list(self.model.heads[-1].parameters()) + list(self.model.final_head[-1].parameters())
        if self.updateqk:
            params = params + list(self.model.headq[-1].parameters()) + list(self.model.headk[-1].parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.optimizer = optimizer

    def pre_train_process(self, index, eval_episodes, info, variant, test_env):
        # warmup phase

        # add the qk module
        self.model.add_qk_module()
        self.model.to(self.device)
        if '_' in index:
            env_name = index.split('_')[0]
        else:
            env_name = index
        self.model.add_task_embeddings(torch.from_numpy(self.task_encoder.encode(self.task_descrption[env_name])).unsqueeze(0).to(self.device))

        params = list(self.model.headq[-1].parameters()) + list(self.model.headk[-1].parameters()) + \
                    list(self.model.heads[-1].parameters())

        optimizer = torch.optim.Adam(params, lr=1e-4)

        self.model.heads[-1].train()
        self.model.headq[-1].train()
        self.model.headk[-1].train()

        for e in range(self.warmup_steps):
            if len(self.masks) > 0:
                batch = self.get_batch(index=index)
                states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
                action_target = torch.clone(actions)

                action_preds = self.model.multi_task_eval(
                    self.masks, env_name, states, actions, rewards, rtg[:,:-1], timesteps, 
                    attention_mask=attention_mask
                )

                act_dim = action_preds.shape[2]
                action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

                loss = self.loss_fn(
                    None, action_preds, None,
                    None, action_target, None,
                )

            else:
                loss = self.warmup_loss(index)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), .25)
            torch.nn.utils.clip_grad_norm_(self.model.headq[-1].parameters(), .25)
            torch.nn.utils.clip_grad_norm_(self.model.headk[-1].parameters(), .25)
            optimizer.step()
        
        # test whether needs to add new module
        if len(self.masks) > 0:
            flag = self.eval_cur_env(eval_episodes, index, info, variant, test_env)
            self.flag = flag
        else:
            self.flag = True
        
        self.model.prev_task[-1] = self.flag
        if self.flag is False:
            self.masks.append({})
        
        # self.model.freeze_headqv()

    
    def train_loop(self, num_steps, env_name=None, iters=0, cur_i=0):
        train_losses = []
        logs = dict()

        if self.flag is False:
            return logs

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
    
    def criterion(self, index, iters):
        """Returns the loss value"""
        batch = self.get_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)

        action_preds = self.model.multi_task_forward(
            self.masks[:iters+1], env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        return loss
    
    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""
        
        # self.fix_biases() # Fix biases after first task
        self.fix_batch_norm() # Fix batch norm mean, var, and params
    
    def eval_cur_env(self, eval_episodes, env_name, info, variant, test_env):
        
        self.model.eval()
        logs = dict()

        # need to sample eval_fn and prompt together 
        self.logger.log(f'Evaluate at task: {env_name}')
        env_id = self.env_name_list.index(env_name)
        self.eval_fns = [eval_episodes(tar, info[env_name], variant, test_env[env_id], env_name) for tar in info[env_name]['env_targets']]
            
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model, info=info, masks=self.masks)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v
        
        total_success_mean = {}
        for k, v in logs.items():
            if 'success_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + '_' + k.split('/')[1].split('_')[1]
                if env not in total_success_mean.keys():
                    total_success_mean[env] = float(v)
                elif total_success_mean[env] < float(v):
                    total_success_mean[env] = float(v)        
        
        self.logger.log(f'current success is {total_success_mean[env_name]}')
        if total_success_mean[env_name] < self.threshold:
            self.logger.log(f'which below the threshold {self.threshold}, adding new module and train from scratch.')
            flag = True
        else:
            self.logger.log(f'which above the threshold {self.threshold}, no new parameters is added.')
            flag = False
        
        return flag
    
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
                    outputs = eval_fn(model, info=info, masks=self.masks)
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
        # self.logger.record_tabular(f'{group}-Total-return-mean', np.mean(total_mean))
        # self.logger.record_tabular(f'{group}-Total-success-mean', np.mean(total_success))
        self.logger.dump_tabular()
        # logs[f'{group}-Total-Return-Mean'] = np.mean(total_mean)
        # logs[f'{group}-Total-Success-Mean'] = np.mean(total_success)

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
                        if mod_name + '.' + name in task:
                            prev_mask |= task[mod_name + '.' + name]
                    
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
                            if mod_name + '.' + name in task:
                                prev_mask |= task[mod_name + '.' + name]

                        curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                        curr_mask = torch.logical_and(curr_mask, ~prev_mask)  # (q & ~p)

                        # Zero non masked weights
                        param_layer *= (curr_mask | prev_mask)
                        
                        mask[mod_name + '.' + name] = curr_mask
        
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
                    # if 'bias' not in name:
                    param_layer.grad *= self.masks[current_task_id][mod_name + '.' + name]
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
                        if mod_name + '.' + name in task:
                            prev_mask |= task[mod_name + '.' + name]

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
                            if mod_name + '.' + name in self.masks[i]:
                                prev_mask |= self.masks[i][mod_name + '.' + name]

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
                        if mod_name + '.' + name in task:
                            prev_mask |= task[mod_name + '.' + name]

                    # Create mask of remaining parameters
                    layer_mask = ~prev_mask
                    mask[mod_name + '.' + name] = layer_mask

        self.masks.append(mask)