# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import transformers

from .trajectory_gpt2 import GPT2Model

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.gelu(input)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class DecisionTransformerPrune(nn.Module):

    def __init__(
            self,
            hidden_size,
            state_dim,
            act_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            env_list=None,
            qkdim=8,
            proj_bias=True,
            recursion=False,
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
        self.recursion = recursion
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        num_embed = 8
        self.env_id_embedding = nn.Embedding(60, num_embed)
        self.env_id_embedding.weight.requires_grad=False

        # self.embed_state = nn.Linear(state_dim + num_embed, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_timestep.weight.requires_grad=False
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.qkdim = qkdim
        self.proj_bias = proj_bias
        self.att_temp = np.sqrt(qkdim)

        self.task_embeddings = []
         
        self.heads = nn.ModuleList()
        self.headq = nn.ModuleList()
        self.headk = nn.ModuleList()

        self.final_head = nn.ModuleList()

        self.prev_task = []
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False
    
    def freeze_headqv(self):
        for param in self.headq.parameters():
            param.requires_grad = False
        for param in self.headk.parameters():
            param.requires_grad = False
    
    def add_head(self):
        self.heads.append(nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if self.action_tanh else []))
        ))
        self.prev_task.append(False)

        self.final_head.append(nn.Sequential(
            init_(nn.Linear(self.hidden_size * 2, self.hidden_size), activate=True), nn.GELU(), nn.LayerNorm(self.hidden_size),
            init_(nn.Linear(self.hidden_size, self.act_dim))
        ))
    
    def add_qk_module(self):
        self.headq.append(nn.Linear(384, self.qkdim, bias=self.proj_bias))
        self.headk.append(nn.Linear(384, self.qkdim, bias=self.proj_bias))
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.headq[-1].apply(init_weights)
        self.headk[-1].apply(init_weights)
    
    def add_task_embeddings(self, embeddings):
        self.task_embeddings.append(embeddings)

    def main_named_parameters(self):
        filtered_params = [(name, param) for name, param in self.named_parameters() if 'head' not in name]
        return filtered_params
    
    def main_named_modules(self):
        filtered_params = [(name, param) for name, param in self.named_modules() if 'head' not in name]
        return filtered_params

    def mask_model(self, masks, reversed=False):
        weights_after_mask=copy.deepcopy(self.state_dict())
        for key, mask in masks.items():
            if key in weights_after_mask:
                if reversed is False:
                    weights_after_mask[key]=weights_after_mask[key]*mask
                else:
                    weights_after_mask[key]=weights_after_mask[key]*(~mask)
        return weights_after_mask
    
    def multi_task_forward(self, masks, env_name, states, actions, rewards, returns_to_go, timesteps, 
                           attention_mask=None, env_ids=None, return_features=False, is_eval=False):
        
        action_preds_prev = []
        prev_feature = []
        original_param = copy.deepcopy(self.state_dict())
        env_ids = self.env_list.index(env_name)

        # first task without mask
        if env_ids == 0:
            action_preds = self.forward(
                env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask
            )[env_ids]

            return action_preds

        prev_task_emd = []

        self.eval()
        prev_model_mask = {}
        
        with torch.no_grad():
            for i, task_mask in enumerate(masks):
                if len(task_mask) == 0:
                    continue
                
                # only use the prev task 
                if i >= env_ids:
                    break

                for k, v in task_mask.items():
                    if k in prev_model_mask:
                        prev_model_mask[k] = (prev_model_mask[k] | v)
                    else:
                        prev_model_mask[k] = v

                weights_after_mask = self.mask_model(prev_model_mask)
                self.load_state_dict(weights_after_mask)

                action_preds, feature = self.forward(
                    env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, return_features=True
                )
                action_preds = action_preds[i]

                if i > 0 and self.recursion:
                    query = self.headq[i](self.task_embeddings[i]) # 1 x dim
                    keys = self.headk[i](torch.cat(prev_task_emd, dim=0)) # i-1 x dim
                    w = (query @ keys.transpose(0, 1)) * (1.0 / self.att_temp)
                    attn = F.softmax(w, dim=-1)
                    concat = torch.cat(prev_feature, dim=0)
                    res = concat * attn.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
                    old_action_preds = torch.sum(res, dim=0)
                    action_preds = torch.cat([old_action_preds, feature], dim=-1)
                    action_preds = self.final_head[i](action_preds)

                action_preds_prev.append(action_preds.unsqueeze(0).detach())
                prev_feature.append(feature.unsqueeze(0).detach())
                prev_task_emd.append(self.task_embeddings[i])

                self.load_state_dict(original_param)

        if is_eval is True:
            self.eval()
        else:
            self.train()
        
        query = self.headq[env_ids](self.task_embeddings[env_ids]) # 1 x dim
        keys = self.headk[env_ids](torch.cat(prev_task_emd, dim=0)) # i-1 x dim

        w = (query @ keys.transpose(0, 1)) * (1.0 / self.att_temp)

        attn = F.softmax(w, dim=-1) # 1 x i-1

        concat = torch.cat(prev_feature, dim=0)
        # concat = self.headv[env_ids](concat)
        res = concat * attn.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        action_preds = torch.sum(res, dim=0)
        
        # if without learning new mask, use all parameters to calculate
        # if have learned new mask, other parameters will be set to zero
        # if eval, only previous and current parameters will be activated
        new_action_preds, feature = self.forward(
            env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, return_features=True
        )


        # action_preds = action_preds + new_action_preds
        action_preds = torch.cat([action_preds, feature], dim=-1)
        action_preds = self.final_head[env_ids](action_preds)

        return action_preds
    
    def multi_task_eval(self, masks, env_name, states, actions, rewards, returns_to_go, timesteps, 
                           attention_mask=None, env_ids=None, return_features=False, is_eval=False):
        action_preds_prev = []
        prev_feature = []
        original_param = copy.deepcopy(self.state_dict())
        env_ids = self.env_list.index(env_name)

        prev_task_emd = []

        prev_model_mask = {}
        with torch.no_grad():
            for i, task_mask in enumerate(masks):
                if len(task_mask) == 0:
                    # not learn new tasks
                    continue
                # only use the prev task 
                if i >= env_ids:
                    break

                for k, v in task_mask.items():
                    if k in prev_model_mask:
                        prev_model_mask[k] = (prev_model_mask[k] | v)
                    else:
                        prev_model_mask[k] = v

                weights_after_mask = self.mask_model(prev_model_mask)
                self.load_state_dict(weights_after_mask)

                action_preds, feature = self.forward(
                    env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, return_features=True
                )
                action_preds = action_preds[i]

                if i > 0 and self.recursion:
                    query = self.headq[i](self.task_embeddings[i]) # 1 x dim
                    keys = self.headk[i](torch.cat(prev_task_emd, dim=0)) # i-1 x dim
                    w = (query @ keys.transpose(0, 1)) * (1.0 / self.att_temp)
                    attn = F.softmax(w, dim=-1)
                    concat = torch.cat(prev_feature, dim=0)
                    res = concat * attn.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
                    old_action_preds = torch.sum(res, dim=0)
                    action_preds = torch.cat([old_action_preds, feature], dim=-1)
                    action_preds = self.final_head[i](action_preds)

                action_preds_prev.append(action_preds.unsqueeze(0).detach())
                prev_feature.append(feature.unsqueeze(0).detach())
                prev_task_emd.append(self.task_embeddings[i])

                self.load_state_dict(original_param)
        
        query = self.headq[env_ids](self.task_embeddings[env_ids]) # 1 x dim
        keys = self.headk[env_ids](torch.cat(prev_task_emd, dim=0)) # i-1 x dim

        w = (query @ keys.transpose(0, 1)) * (1.0 / self.att_temp)
        if is_eval is True:
            index = w.topk(k=1, dim=1)[1]
            attn = torch.zeros_like(w, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        else:
            attn = F.softmax(w, dim=-1) # 1 x i-1

        concat = torch.cat(action_preds_prev, dim=0)
        # concat = self.headv[env_ids](concat)
        res = concat * attn.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        action_preds = torch.sum(res, dim=0)

        return action_preds


    def forward(self, env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, env_ids=None, return_features=False):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # task identity
        # env_ids = [self.env_list.index(env_name)] * batch_size
        # env_ids_embeddings = self.env_id_embedding(torch.Tensor(env_ids).long().to(device=states.device)).unsqueeze(1).repeat(1, seq_length, 1)
        # states = torch.cat([states, env_ids_embeddings], dim=-1)
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
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
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
            return y, x[:, 1]
        else:
            return y

    def get_action(self, env_name, states, actions, rewards, returns_to_go, timesteps, info, masks, **kwargs):

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
        
        num_head = len(self.heads)
        env_id = self.env_list.index(env_name)

        if masks is not None and env_id < num_head:
            if self.prev_task[env_id] is False:
                # only use prev task to calculate current task
                action_preds = self.multi_task_eval(masks[:env_id+1], env_name, states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, is_eval=True)
            else:
                original_param = copy.deepcopy(self.state_dict())
                action_preds = self.multi_task_forward(masks[:env_id+1], env_name, states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, is_eval=True) 
                self.load_state_dict(original_param)         
        else:
            id = min(env_id, num_head-1)
            action_preds = self.forward(
                env_name, states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)[id]

        return action_preds[0,-1]
