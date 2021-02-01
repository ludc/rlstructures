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

    def call_bp(self,trajectories,agent_info,t,last_call_state):
        assert not self.require_history()

        tslice=trajectories.temporal_slice(t)
        if last_call_state is None:
            assert t==0
            last_call_state=tslice.truncate_key("agent_state")
        else:
            last_call_state=last_call_state.truncate_key("_agent_state/")

        observation=tslice.truncate_key("observation")
        action,n_agent_state=self.__call__(state,observation,agent_info,None)

        output=(last_call_state.prepend_key("agent_state/")
                +action.prepend_key("action/")
                +n_agent_state.prepend_key("_agent_state/")
        )
        return output

    def update(self, info):
        raise NotImplementedError

    def close(self):
        pass

    def get_default_agent_info(self,batch_size):
        raise NotImplementedError

def replay_agent(agent,trajectories,info,function_name="call_bp"):
    #TODO: Computation is made on all batches, could use the mask to reduce the amount of computations
    agent_info=info.truncate_key("agent_info/")

    T=trajectories.lengths.max().item()
    f=getattr(agent,function_name)
    tdt=None
    assert not agent.require_history(),"History not implemented in replay_agent"

    output=f(trajectories,agent_info,0,None)
    variables={}
    for k in output.keys():
        s=output[k].size()
        t=torch.zeros(s[0],T,*s[1:],dtype=output[k].dtype)
        t[:,0]=output[k]
        variables[k]=t

    for t in range(1,T):
        output=f(trajectories,agent_info,t,output)
        for k in output.keys():
            variables[k][:,t]=output[k]
    tdt=TemporalDictTensor(variables,lengths=trajectories.lengths.clone())

    return tdt
