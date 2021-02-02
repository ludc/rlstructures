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

    def call_replay(self,trajectories,info,t,state):
        assert not self.require_history()
        if state is None:
            assert t==0
            state=info.truncate_key("agent_state/")
        agent_info=info.truncate_key("agent_info/")
        tslice=trajectories.temporal_index(t)
        observation=tslice.truncate_key("observation/")
        action,state=self.__call__(state,observation,agent_info,None)

        return action,state

    def initial_state(self,agent_info,B):
        raise NotImplementedError

    def update(self, info):
        raise NotImplementedError

    def close(self):
        pass

def replay_agent_stateless(agent,trajectories,info,replay_method_name):
    """
    Replay transitions all in one
    returns a TDT
    """
    f=getattr(agent,replay_method_name)
    return f(trajectories,info)

def replay_agent(agent,trajectories,info,replay_method_name="call_replay"):
    """
    Replay transitions one by one in the temporal order, passing a state between each call
    returns a TDT
    """
    T=trajectories.lengths.max().item()
    f=getattr(agent,replay_method_name)

    output,state=f(trajectories,info,0,None)
    variables={}

    for k in output.keys():
        s=output[k].size()
        t=torch.zeros(s[0],T,*s[1:],dtype=output[k].dtype)
        t[:,0]=output[k]
        variables[k]=t

    for t in range(1,T):
        output,state=f(trajectories,info,t,state)
        for k in output.keys():
            variables[k][:,t]=output[k]
    tdt=TemporalDictTensor(variables,lengths=trajectories.lengths.clone())

    return tdt
