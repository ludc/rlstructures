
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Agents/Policies in RLStructures
#
# An agent is the (only) abstraction need to allow RLStructures to collect interactions at scale.
# * An Agent class represents a policy (or *multiple policies*)
# * An Agent may include (or not) one or multiple pytorch Module, but it is not mandatory
# * An Agent is stateless, and implements a **__call__** methods
# * The **__call__(agent_state,observation,agent_info)** methods takes as an input:
# * * The state of the agent as time t-1 (as a DictTensor). The state will store all the informations needed to continue the execution of the agent
# * * The observation provided by the *rlstructures.VecEnv* environments
# * * * Note that agent_state.n_elems()==observation.n_elems()
# * * an *agent_info* DictTensor that is an additional information that will be controlled by the user
# * * * For instace, *agent_info* will contains a flag telling if the agent has to be stochastic or deterministic
# * An Agent implements an initial_state method to compute its initial internal state


# As an output, the **__call__** method returns *(old_state,action,new_state)* where:
# * *action* is the action outputed by the agent as a DictTensor. Note that *action.n_elems()==observation.n_elems()*. This information will be transmitted to the environment through the *env.step* method.
# * *new_state* is the update of the state of the agent. This new state is the information transmitted to the Agent at the next call of the agent



from rlstructures import RL_Agent,DictTensor
import torch

class UniformAgent(RL_Agent):
    def __init__(self,n_actions):
        super().__init__()
        self.n_actions=n_actions

    def initial_state(self,agent_info,B):
        return DictTensor({"timestep":torch.zeros(B).long()})

    def __call__(self,state,observation,agent_info=None,history=None):
        B=observation.n_elems()

        scores=torch.randn(B,self.n_actions)
        probabilities=torch.softmax(scores,dim=1)
        actions=torch.distributions.Categorical(probabilities).sample()
        new_state=DictTensor({"timestep":state["timestep"]+1})
        return DictTensor({"action":actions}),new_state
