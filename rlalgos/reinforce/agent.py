#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import time
import torch.nn as nn
from rlstructures import DictTensor,masked_tensor,masked_dicttensor
from rlstructures import RL_Agent

class ReinforceAgent(RL_Agent):
    def __init__(self,model=None, n_actions=None):
        super().__init__()
        self.model = model
        self.n_actions = n_actions

    def update(self,  state_dict):
        print(self)
        self.model.load_state_dict(state_dict)
        print(state_dict)
        print(self.model.state_dict())
        print("====")

    def require_history(self):
        return False

    def initial_state(self,agent_info,B):
        return DictTensor({})

    def __call__(self, state,observation,agent_info=None,history=None):
        """
        Executing one step of the agent
        """
        assert state.empty()

        B = observation.n_elems()
        action_proba = self.model.action_model(observation["frame"])
        baseline = self.model.baseline_model(observation["frame"])
        dist = torch.distributions.Categorical(action_proba)
        action_sampled = dist.sample()

        action_max = action_proba.max(1)[1]
        smask=agent_info["stochastic"].float().to(action_max.device)
        action=masked_tensor(action_max,action_sampled,smask)

        new_state = DictTensor({})

        agent_do = DictTensor(
            {"action": action, "action_probabilities": action_proba, "baseline":baseline}
        )

        return agent_do, new_state

class Model(nn.Module):
    def __init__(self,action_model,baseline_model):
        super().__init__()
        self.action_model=action_model
        self.baseline_model=baseline_model

class ActionModel(nn.Module):
    """ The model that computes one score per action
    """
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)


    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        probabilities_actions = torch.softmax(score_actions,dim=-1)
        return probabilities_actions

class BaselineModel(nn.Module):
    """ The model that computes V(s)
    """
    def __init__(self, n_observations, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)


    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        critic = self.linear2(z)
        return critic
