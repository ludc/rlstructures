#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import logging
from rlstructures import DictTensor,TemporalDictTensor
import torch

class S_Agent:
    def __init__(self):
        pass

    def require_history(self):
        return False

    def __call__(self, state:DictTensor, input:DictTensor,agent_info:DictTensor,history:TemporalDictTensor = None):
        raise NotImplementedError

    def call_bp(self,state, observation,action,agent_info,history=None):
        raise NotImplementedError

    def update(self, info):
        raise NotImplementedError

    def close(self):
        pass

    def get_default_agent_info(self,batch_size):
        raise NotImplementedError

def replay_agent(self,agent,trajectories,info,function_name="call_bp"):
    agent_info=info.truncate_key("agent_info/")
    env_info=info.truncate_key("env_info/")
    T=trajectories.lengths.max().item()
    tslice=trajectories.temporal_index(0)
    agent_state=tslice.truncate_key("agent_state/")
    observation=tslice.truncate_key("observation/")
    action=tslice.truncate_key("action/")
    f=getattr(agent,function_name)
    agent_states=[agent_state]
    actions=[]
    for t in range(T):
        agent_state,action=f(agent_state,observation,action,agent_info)
        agent_states.append(agent_state)
        actions.append(action)
