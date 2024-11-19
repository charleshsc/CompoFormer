import torch
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                num_of_fisher=1, lamb=5000, alpha=0.5, fi_sampling_type='max_pred',
                fi_num_samples=-1, **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)

        self.num_of_fisher = num_of_fisher
        self.lamb = lamb
        self.alpha = alpha
        self.sampling_type = fi_sampling_type
        self.num_samples = fi_num_samples

        # In all cases, we only keep importance weights for the model, but not for the heads.
        # Store current parameters as the initial parameters before first task starts
        self.older_params= {n: p.clone().detach() for n, p in self.model.main_named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.main_named_parameters()
                       if p.requires_grad}

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--num_of_fisher', default=1, type=int)
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument('--lamb', default=5000, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Define how old and new fisher is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='EWC alpha (default=%(default)s)')
        parser.add_argument('--fi-sampling-type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'],
                            help='Sampling type for Fisher information (default=%(default)s)')
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')

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


    def compute_fisher_matrix_diag(self, index):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.main_named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        self.model.train()

        for i in range(self.num_of_fisher):
            
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
            # Accumulate all gradients from loss with regularization
            for n, p in self.model.main_named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(action_target)
        # Apply mean across all samples
        n_samples = self.num_of_fisher * len(action_target)
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.main_named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(index)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            if self.alpha == -1:
                raise NotImplementedError('alpha should not be -1')
            else:
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n])
    
    def criterion(self, index, iters):
        """Returns the loss value"""
        loss = super().criterion(index, iters)

        if iters > 0:
            loss_reg = 0
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.model.main_named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * loss_reg

        return loss