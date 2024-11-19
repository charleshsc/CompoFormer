# import d4rl
from ast import parse
import numpy as np
import torch
import os
import time
import pathlib
import argparse
import pickle
import random
import sys
from tqdm import trange

import itertools
import copy

from dt.decision_transformer import DecisionTransformer
from dt.decision_transformer_lora import DecisionTransformerLora
from dt.decision_transformer_grow import DecisionTransformerGrow
from dt.decision_transformer_prune import DecisionTransformerPrune
from dt.seq_trainer import SequenceTrainer
from dt.utils import get_env_list, report_parameters, get_batch, eval_episodes
from dt.utils import process_info, load_meta_data
import dt.Metric as metric
from dt.approach.incremental_learning import Inc_Learning_Appr

from collections import namedtuple
import json, pickle, os
from logger import setup_logger, logger

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def experiment_mix_env(
        variant,
        appr_args,
        Appr,
):
    device = variant['device']
    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    seed = variant['seed']
    set_seed(variant['seed'])
    env_name_ = variant['env']
    
    ######
    # construct train and test environments
    ######
    data_save_path = variant['data_path']
    save_path = variant['save_path']
    timestr = time.strftime("%y%m%d-%H%M%S")

    task_config = os.path.join(variant['config_path'], "MetaWorld/task.json")
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    
    if 'cw10' in variant['prefix_name'] and 'seed1' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.cw10_task_list_seed1, task_config.cw10_task_list_seed1
    elif 'cw10' in variant['prefix_name'] and 'seed2' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.cw10_task_list_seed2, task_config.cw10_task_list_seed2
    elif 'cw10' in variant['prefix_name'] and 'seed3' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.cw10_task_list_seed3, task_config.cw10_task_list_seed3
    elif 'cw10' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.cw10_task_list, task_config.cw10_task_list
    elif 'cw20' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.cw20_task_list, task_config.cw20_task_list
    else:
        raise NotImplementedError("please use the cw10")
    n_task = len(train_env_name_list)

    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, device, total_env='metaworld_test', seed=seed)
    # train and test share the env
    info = copy.deepcopy(test_info)

    group_name = variant['prefix_name']
    if variant['continual_method']:
        continual_method = variant['continual_method']
    else:
        continual_method = 'base'
    exp_prefix = f'{env_name_}-{seed}-{timestr}'
    if variant['no_r']:
        exp_prefix += '_NO_R'
    save_path = os.path.join(save_path, group_name+'-'+continual_method+'-'+exp_prefix)
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    setup_logger(exp_prefix, variant=variant, variant2=appr_args, log_dir=save_path)

    logger.log(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    
    ######
    # process train and test datasets
    ######
    # load training dataset 
    optimal = not variant['suboptimal']
    trajectories_list = load_meta_data(train_env_name_list, data_save_path, optimal)

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, pct_traj, variant, logger)
    # share the info
    test_info = copy.deepcopy(info)

    ######
    # construct dt model and trainer
    ######
    state_dim = info[train_env_name_list[0]]['state_dim']
    act_dim = info[train_env_name_list[0]]['act_dim']
    if variant['continual_method'] is not None and 'lora' in variant['continual_method']:
        Model = DecisionTransformerLora
    elif variant['continual_method'] is not None and 'grow' in variant['continual_method']:
        Model = DecisionTransformerGrow
    elif variant['continual_method'] is not None and 'prune' in variant['continual_method']:
        Model = DecisionTransformerPrune
    else:
        Model = DecisionTransformer
    model = Model(
        hidden_size=variant['embed_dim'],
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=1000,
        env_list=train_env_name_list,
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        recursion=variant['recursion'],
        **appr_args
    )
    model = model.to(device=device)
    report_parameters(model, logger)

    trainer = Appr(
        model=model,
        batch_size=batch_size,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        device=device,
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
        warmup_steps=variant['warmup_steps'], 
        warmup_lr=variant['warmup_lr'],
        eval_fns=None,
        get_batch=get_batch(trajectories_list, info, variant, train_env_name_list),
        logger=logger,
        n_task=n_task,
        env_name_list=test_env_name_list,
        **appr_args
    )

    ######
    # start training
    ######

    ## First evaluate the performance with inital parameter
    model.add_head()
    test_eval_logs = trainer.eval_iteration_metaworld(
        eval_episodes, test_env_name_list, test_info, variant, test_env_list, 
        iter_num=0, group='init test')
    init_performance = test_eval_logs['total_success']

    total_success = []
    log_performance = []
    for iter, env_name in enumerate(train_env_name_list):
        if iter > 0:
            trainer.model.add_head()
            trainer.model = model.to(device=device)
        trainer.pre_train_process(env_name, eval_episodes=eval_episodes, info=test_info, variant=variant, test_env=test_env_list)
        trainer.set_optimizer()
        
        num_of_logs = int(variant['num_iters_per_env'] / variant['num_iters_per_log'])
        for i in range(num_of_logs):
            logs = trainer.train_loop(
                num_steps=variant['num_iters_per_log'], 
                env_name = env_name,
                iters = iter,
                cur_i=i,
            )
            
            # log training information
            logger.record_tabular('Env Name', env_name) 
            for key, value in logs.items():
                logger.record_tabular(key, value)
            logger.dump_tabular() 
            group='test-seperate'
            
            # evaluate test
            iter_num = iter * variant['num_iters_per_env'] + (i+1) * variant['num_iters_per_log']
            # Not training, we directly use the performance of iteration 0.
            if 'zeta' in variant['continual_method'] and trainer.flag == False:
                if i == 0:
                    test_eval_logs = trainer.eval_iteration_metaworld(
                        eval_episodes, test_env_name_list, test_info, variant, test_env_list, 
                        iter_num=iter_num, group=group)
                
                    current_epoch_success = test_eval_logs['total_success']
                    log_performance.append(np.mean(current_epoch_success))
                    total_success.append(current_epoch_success)
                else:
                    log_performance.append(log_performance[-1])
                
            else:
                test_eval_logs = trainer.eval_iteration_metaworld(
                    eval_episodes, test_env_name_list, test_info, variant, test_env_list, 
                    iter_num=iter_num, group=group)
                
                current_epoch_success = test_eval_logs['total_success']
                log_performance.append(np.mean(current_epoch_success))
            
                if i + 1 == num_of_logs:
                    total_success.append(current_epoch_success)
        
        trainer.post_train_process(env_name)

        logger.log(str(total_success))
    
    logger.log('------------ Final Performance -----------------')
    logger.log('------------------------------------------------')
    logger.log(f'Final performance is {metric.calculate_mean_performance(total_success)}')
    logger.log(f'Final epoch performance is {metric.calculate_mean_performance(total_success)[-1]}')
    logger.log(f'Final forgetting is {metric.calculate_forgetting(total_success)}')
    logger.log(f'Final backward transfer is {metric.calculate_backward_transfer(total_success)}')
    logger.log(f'Final forward transfer is {metric.calculate_forward_transfer(total_success, init_performance)}')
    logger.log('------------------------------------------------')


    trainer.save_model(env_name=args.env, final_performance=log_performance, postfix='iter_'+str(iter + 1),  folder=save_path)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['MetaWorld'], default='MetaWorld') 
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--config_path', type=str, default='./config')
    parser.add_argument('--prefix_name', type=str, default='cw10')
    parser.add_argument('--suboptimal', action='store_true', default=False)
    parser.add_argument('--continual_method', type=str, default=None)
    parser.add_argument('--recursion', action='store_true', default=False)

    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_lr', type=float, default=1e-4)
    parser.add_argument('--num_eval_episodes', type=int, default=10) 
    parser.add_argument('--max_iters', type=int, default=1000000) 
    parser.add_argument('--num_iters_per_env', type=int, default=50000)
    parser.add_argument('--num_iters_per_log', type=int, default=10000)
    parser.add_argument('--test_eval_interval', type=int, default=5000)
    parser.add_argument('--test_eval_seperate_interval', type=int, default=10000)


    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load_path', type=str, nargs='+', default=None) # choose a model when in evaluation mode
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--data_path', type=str, default='./MT50/dataset')
    
    args, extra_args = parser.parse_known_args()

    if args.continual_method is not None:
        import importlib
        Appr = getattr(importlib.import_module(name='dt.approach.' + args.continual_method), 'Appr')
    else:
        Appr = Inc_Learning_Appr
    
    appr_args, extra_args = Appr.extra_parser(extra_args)

    
    experiment_mix_env(variant=vars(args), appr_args=vars(appr_args), Appr=Appr)