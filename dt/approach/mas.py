import torch
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                num_of_importance=1, lamb=1, alpha=0.5, **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)

        self.num_of_importance = num_of_importance
        self.lamb = lamb
        self.alpha = alpha
        
        # In all cases, we only keep importance weights for the model, but not for the heads.
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in self.model.main_named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.main_named_parameters()
                           if p.requires_grad}
    
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--num_of_importance', default=1, type=int)
        # Eq. 3: lambda is the regularizer trade-off -- In original code: MAS.ipynb block [4]: lambda set to 1
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off  (default=%(default)s)')
        # Define how old and new importance is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='MAS alpha (default=%(default)s)')
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
    
    # Section 4.1: MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, index):
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.main_named_parameters()
                      if p.requires_grad}

        # Do forward and backward pass to accumulate L2-loss gradients
        self.model.train()
        for i in range(self.num_of_importance):
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

            self.optimizer.zero_grad()
            loss.backward()
            # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
            for n, p in self.model.main_named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(action_target)
        # Eq. 2: divide by N total number of samples
        n_samples = self.num_of_importance * len(action_target)
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance
    
    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.main_named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_importance = self.estimate_parameter_importance(index)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.importance.keys():            
            # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
            self.importance[n] = self.alpha * self.importance[n] + (1 - self.alpha) * curr_importance[n]

    def criterion(self, index, iters):
        """Returns the loss value"""
        loss = super().criterion(index, iters)

        if iters > 0:
            loss_reg = 0
            # Eq. 3: memory aware synapses regularizer penalty
            for n, p in self.model.main_named_parameters():
                if n in self.importance.keys():
                    loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * loss_reg
        
        return loss
    


