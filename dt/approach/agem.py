import torch
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr
import random


class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                buffer_size=20, add_buffer_times_per_env=5,
                **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)

        self.buffer_size = buffer_size
        self.add_buffer_times_per_env =add_buffer_times_per_env
        
        self.buffers = []
        self.buffer_iter = 0
        self.reference_gradients = torch.empty(0)

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--num_of_fisher', default=1, type=int)
        parser.add_argument('--buffer_size', default=20, type=int, required=False,)
        parser.add_argument('--add_buffer_times_per_env', default=5, type=int, required=False,)
        

        return parser.parse_known_args(args)

    def set_optimizer(self):
        """Returns the optimizer"""
        embed_params = [param for name, param in self.model.main_named_parameters() if param.requires_grad]
        params = embed_params + list(self.model.heads[-1].parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.optimizer = optimizer


    def before_training_iteration(self):
        """
        Compute reference gradient on memory sample.
        """

        if len(self.buffers) > 0:
            self.model.train()
            self.optimizer.zero_grad()
            mb = self.sample_from_memory()

            states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = mb
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
            loss.backward()

            # gradient can be None for some head on multi-headed models
            reference_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=self.device)
                )
                for n, p in self.model.main_named_parameters()
            ]
            self.reference_gradients = torch.cat(reference_gradients_list)
            self.optimizer.zero_grad()
    
    def train_iteration(self, index, iters):
        """
        Project gradient based on reference gradients
        """
        self.before_training_iteration()

        self.model.train()

        loss = self.criterion(index, iters)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        if len(self.buffers) > 0:
            current_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=self.device)
                )
                for n, p in self.model.main_named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                )
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for n, p in self.model.main_named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                    count += n_param

        self.optimizer.step()

        return loss.detach().cpu().item()

    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""

        for i in range(self.add_buffer_times_per_env):
            batch = self.get_batch(index=index)
            self.buffers.append(batch)
        
        removed_els = len(self.buffers) - self.buffer_size
        if removed_els > 0:
            indices = list(range(len(self.buffers)))
            random.shuffle(indices)
            self.buffers = [self.buffers[i] for i in indices[:self.buffer_size]]
    
    def sample_from_memory(self):
        if self.buffer_iter >= len(self.buffers):
            self.buffer_iter = 0
        return self.buffers[self.buffer_iter]