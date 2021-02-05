from rlstructures import logging
from rlstructures.env_wrappers import GymEnv, GymEnvInf
from rlstructures.tools import weight_init
from rlstructures.batchers import Batcher,EpisodeBatcher
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from e_rlalgos.dqn.agent import QAgent, QMLP, DQMLP, DuelingCnnDQN, CnnDQN
import gym
from gym.wrappers import TimeLimit
from e_rlalgos.dqn.duelling_dqn import DQN
from e_rlalgos.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math
import itertools

def create_env(n_envs, mode="train",max_episode_steps=None, seed=None,**args):


    if mode=="train":
        envs=[]
        for k in range(n_envs):
            e = make_atari(args["environment/env_name"])
            e = wrap_deepmind(e)
            e = wrap_pytorch(e)
            envs.append(e)
        return GymEnvInf(envs, seed)
    else:
        envs=[]
        for k in range(n_envs):
            e = make_atari(args["environment/env_name"])
            e = wrap_deepmind(e)
            e = wrap_pytorch(e)
            envs.append(e)
        return GymEnv(envs, seed)

def create_agent(n_actions=None, model=None, ):
    return QAgent(model=model, n_actions=n_actions)

class Experiment(DQN):
    def __init__(self, config, create_env, create_agent):
        super().__init__(config, create_env, create_agent)

    def _create_model(self):
        if (self.config["use_duelling"]):
            module = DuelingCnnDQN(self.obs_shape,self.n_actions)
        else:
            module = CnnDQN(self.obs_shape,self.n_actions)
        #module.apply(weight_init)
        return module

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def generate_grid_search(config):
    c = {}
    for k in config:
        if not isinstance(config[k], list):
            c[k] = [config[k]]
        else:
            c[k] = config[k]
    return product_dict(**c)

def generate(common,specifics):
    gs=list(generate_grid_search(common))
    print("Common ",len(gs)," Specific ",len(specifics))
    r=[]
    for s in specifics:
        for c in gs:
            cc=copy.deepcopy(c)
            for k in s:
                cc[k]=s[k]
            r.append(cc)
    print("== Total ",len(r))
    return r

if __name__=="__main__":
    #We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    config={"environment/env_name": "PongNoFrameskip-v4",
            "n_envs": 1,
            "max_episode_steps": 10000,
            "discount_factor": 0.99,
            "epsilon_greedy_max": 0.5,
            "epsilon_greedy_min": 0.1,
            "epsilon_min_epoch": 1000000,
            "replay_buffer_size": 100000,
            "n_batches": 32,
            "update_target_epoch":1000,
            "tau": 0.005,
            "initial_buffer_epochs": 1,
            "qvalue_epochs": 10,
            "batch_timesteps": 10,
            "use_duelling": False,
            "use_double":False,
            "lr": 0.0001,
            "n_processes": 1,
            "n_evaluation_processes": 4,
            "verbose": True,
            "n_evaluation_envs": 4,
            "time_limit": 28800,
            "env_seed": 48,
            "clip_grad": 0.0,
            "learner_device": "cuda",
            "logdir":"./results",
            "update_target_epoch":1000

    }
    import sys
    import os
    print(len(sys.argv))
    if len(sys.argv)==2:

        print("opening ",sys.argv[1])
        _file = open(sys.argv[1], "r")
        c=_file.read()
        _file.close()
        common,specifics=eval(c)
        r=generate(common,specifics)
        if os.environ.get("SLURM_ARRAY_TASK_ID")==None:
            exit()
        config=r[int(os.environ.get("SLURM_ARRAY_TASK_ID"))]
        logdir=config["logdir"]
        config["logdir"]=logdir+"/"+str(int(os.environ.get("SLURM_ARRAY_TASK_ID")))

    exp=Experiment(config,create_env,create_agent)
    exp.run()
