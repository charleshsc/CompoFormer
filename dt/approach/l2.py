import torch
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                lamb=5000, **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)
        
        self.lamb = lamb

        # In all cases, we only keep importance weights for the model, but not for the heads.
        # Store current parameters as the initial parameters before first task starts
        self.older_params= {n: p.clone().detach() for n, p in self.model.main_named_parameters() if p.requires_grad}
        

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument('--lamb', default=5000, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')

        return parser.parse_known_args(args)

    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.main_named_parameters() if p.requires_grad}
    
    def criterion(self, index, iters):
        """Returns the loss value"""

        loss = super().criterion(index, iters)

        if iters > 0:
            loss_reg = 0
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.model.main_named_parameters():
                if p.requires_grad:
                    loss_reg += torch.sum((p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * loss_reg

        return loss