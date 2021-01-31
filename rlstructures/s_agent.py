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

    def call_bp(self,state, observation,action,agent_info,trajectories,history):
        return self.__call__(state,observation,agent_info,history)

    def update(self, info):
        raise NotImplementedError

    def close(self):
        pass

    def get_default_agent_info(self,batch_size):
        raise NotImplementedError

def replay_agent(agent,trajectories,info,function_name="call_bp"):
    #TODO: Computation is made on all batches, could use the mask to reduce the amount of computations
    agent_info=info.truncate_key("agent_info/")
    env_info=info.truncate_key("env_info/")

    tdt=trajectories.clone()

    T=trajectories.lengths.max().item()
    tslice=trajectories.temporal_index(0)
    agent_state=tslice.truncate_key("agent_state/")
    f=getattr(agent,function_name)
    actions=[]
    assert not agent.require_history(),"History not implemented in replay_agent"

    for t in range(T):
        tslice=trajectories.temporal_index(t)
        observation=tslice.truncate_key("observation/")
        action=tslice.truncate_key("action/")

        for k in agent_state.keys():
            tdt.variables["agent_state/"+k][:,t]=agent_state[k]


        action,agent_state=f(agent_state,observation,action,agent_info,history=tdt,trajectories=trajectories)

        if t==0:
            for k in action.keys():
                if not "action/"+k in tdt.variables:
                    s=action[k].size()
                    nt=torch.zeros(s[0],T,*s[1:],dtype=action[k].dtype)
                    tdt.set("action/"+k,nt)

        #Copy of agent_state
        for k in agent_state.keys():
            tdt.variables["_agent_state/"+k][:,t]=agent_state[k]
        for k in action.keys():
            tdt.variables["action/"+k][:,t]=action[k]
    return tdt
