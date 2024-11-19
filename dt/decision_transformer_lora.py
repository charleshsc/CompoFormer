# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn

import transformers

from .trajectory_gpt2_lora import GPT2Model

class DecisionTransformerLora(nn.Module):

    def __init__(
            self,
            hidden_size,
            state_dim,
            act_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            env_list=None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.max_ep_len = max_ep_len
        self.action_tanh = action_tanh
        self.env_list = env_list
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        num_embed = 8
        self.env_id_embedding = nn.Embedding(60, num_embed)
        self.env_id_embedding.weight.requires_grad=False

        self.embed_state = nn.Linear(state_dim + num_embed, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
         
        # self.predict_return = nn.Linear(hidden_size, 1)
        # self.predict_state = nn.Linear(hidden_size, state_dim)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        self.heads = nn.ModuleList()
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False
        
    def add_lora(self):
        self.transformer.add_lora_part()
    
    def add_head(self):
        self.heads.append(nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if self.action_tanh else []))
        ))

    def main_named_parameters(self):
        filtered_params = [(name, param) for name, param in self.named_parameters() if 'heads' not in name]
        return filtered_params
    
    def main_named_modules(self):
        filtered_params = [(name, param) for name, param in self.named_modules() if 'heads' not in name]
        return filtered_params


    def forward(self, env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, env_ids=None, return_features=False):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # task identity
        env_ids = [self.env_list.index(env_name)] * batch_size
        env_ids_embeddings = self.env_id_embedding(torch.Tensor(env_ids).long().to(device=states.device)).unsqueeze(1).repeat(1, seq_length, 1)
        states = torch.cat([states, env_ids_embeddings], dim=-1)
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

         # we feed in the input embeddings (not word indices as in NLP) to the model
        id = min(self.env_list.index(env_name), len(self.heads)-1)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            task_id=id,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)


        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        # return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        # action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state
        
        y = []
        for head in self.heads:
            y.append(head(x[:,1])[:, -seq_length:, :])
        
        if return_features:
            return y, x
        else:
            return y

    def get_action(self, env_name, states, actions, rewards, returns_to_go, timesteps, info, **kwargs):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        env_id = self.env_list.index(env_name)
        num_head = len(self.heads)
        id = min(env_id, num_head-1)
        action_preds = self.forward(
            env_name, states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)[id]

        return action_preds[0,-1]
