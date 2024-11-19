import torch
from copy import deepcopy
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    def __init__(self, model, batch_size, loss_fn, device,
                lr=1e-4, weight_decay=1e-4, warmup_steps=0, warmup_lr=1e-4,
                eval_fns=None, get_batch=None, logger=None, n_task=0,
                lamb=1, T=2, **kwargs):

        super(Appr, self).__init__(model, batch_size, loss_fn, device,
                lr, weight_decay, warmup_steps, warmup_lr, eval_fns, get_batch, logger, n_task)

        self.lamb = lamb
        self.T = T

        self.model_old = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
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

    def post_train_process(self, index):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
    
    def criterion(self, index, iters):
        """Returns the loss value"""
        batch = self.get_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)

        y = self.model.forward(
            env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask
        )
        action_preds = y[-1]

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        if self.model_old is not None and len(y) > 1:
            y_old = self.model_old.forward(
                env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask
            )
            
            loss_reg = 0
            for i in range(len(y)-1):
                y_prev = y[i].reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                y_prev_old = y_old[i].reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

                loss_reg += self.loss_fn(
                    None, y_prev, None,
                    None, y_prev_old, None,
                )


            loss += self.lamb * loss_reg

        return loss