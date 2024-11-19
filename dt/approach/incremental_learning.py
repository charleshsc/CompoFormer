import torch
import numpy as np
from argparse import ArgumentParser
import time

class Inc_Learning_Appr:
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0, **kwargs):

        self.model = model
        self.optimizer = None
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.n_task = n_task

        self.eval_fns = [] if eval_fns is None else eval_fns
        self.get_batch = get_batch

        self.start_time = time.time()
        self.logger = logger

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)
    
    def set_optimizer(self):
        """Returns the optimizer"""
        params = self.model.parameters()
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.optimizer = optimizer
    
    def pre_train_process(self, index, **kwargs):
        # warmup phase
        if self.warmup_steps > 0:

            optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)

            for e in range(self.warmup_steps):
                self.model.heads[-1].train()
                loss = self.warmup_loss(index)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), .25)
                optimizer.step()

    
    def train_loop(self, num_steps, env_name=None, iters=0, cur_i=0):
        train_losses = []
        logs = dict()

        train_start = time.time()

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
        pass
    
    def criterion(self, index, iters):
        """Returns the loss value"""
        batch = self.get_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)

        action_preds = self.model.forward(
            env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask
        )[-1]

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        return loss
    
    def warmup_loss(self, index):
        """Returns the loss value"""
        batch = self.get_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)

        action_preds = self.model.forward(
            env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask
        )[-1]

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        return loss
    
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
            
            self.model.eval()
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

 
    def save_model(self, env_name, final_performance, postfix, folder):

        model_name = '/model_' + postfix
        saved = {
            'model': self.model.state_dict(),
            'final_per': final_performance,
        }
        torch.save(saved, folder+model_name)  # model save
        self.logger.log('model saved to ' + folder+model_name)