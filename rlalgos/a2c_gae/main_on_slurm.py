#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from rlstructures.env_wrappers import GymEnv, GymEnvInf
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from rlalgos.a2c_gae.atari_agent import AtariAgent,ActionModel,CriticModel,Model
from rlalgos.a2c_gae.a2c import A2C
import gym
from gym.wrappers import TimeLimit
from gym import ObservationWrapper
from rlalgos.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch
import itertools

def create_env(n_envs, env_name,max_episode_steps=None, seed=None,**args):
        envs=[]
        for k in range(n_envs):
            e = make_atari(env_name)
            e = wrap_deepmind(e)
            e = wrap_pytorch(e)
            envs.append(e)
        return GymEnv(envs, seed)

def create_train_env(n_envs, env_name,max_episode_steps=None, seed=None,**args):
        envs=[]
        for k in range(n_envs):
            e = make_atari(env_name)
            e = wrap_deepmind(e)
            e = wrap_pytorch(e)
            envs.append(e)
        return GymEnvInf(envs, seed)


def create_agent(model,n_actions=1):
    return AtariAgent(model=model, n_actions=n_actions)

class Experiment(A2C):
    def __init__(self, config, create_train_env, create_env, create_agent):
        super().__init__(config, create_train_env, create_env, create_agent)


    def _create_model(self):
        am=ActionModel(self.obs_shape, self.n_actions)
        cm=CriticModel(self.obs_shape)
        model=Model(am,cm)
        model.apply(weight_init)
        return model

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
