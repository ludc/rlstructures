#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


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
            e = make_atari(args["environment/env_name"], max_episode_steps=max_episode_steps)
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
        module = DuelingCnnDQN(self.obs_shape,self.n_actions)
        module.apply(weight_init)
        return module

def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__=="__main__":
    #We use spawn mode such that most of the environment will run in multiple processes
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    config={"environment/env_name": "PongNoFrameskip-v4",
            "n_envs": 4,
            "max_episode_steps": 100,
            "discount_factor": 0.99,
            "epsilon_greedy_max": 0.5,
            "epsilon_greedy_min": 0.01,
            "replay_buffer_size": 100000,
            "n_batches": 32,
            "tau": 0.005,
            "initial_buffer_epochs": 100,
            "qvalue_epochs": 1,
            "batch_timesteps": 1,
            "use_duelling": True,
            "lr": 0.0003,
            "n_processes": 4,
            "n_evaluation_processes": 4,
            "verbose": True,
            "n_evaluation_envs": 16,
            "time_limit": 180000,
            "env_seed": 42,
            "clip_grad": 10.0,
            "learner_device": "cpu",
            "logdir":"./results"
    }
    exp=Experiment(config,create_env,create_agent)
    exp.run()
