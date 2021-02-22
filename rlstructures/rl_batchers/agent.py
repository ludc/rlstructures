#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import DictTensor, TemporalDictTensor, Trajectories
import torch


class RL_Agent:
    def __init__(self):
        pass

    def require_history(self):
        return False

    def initial_state(self, agent_info: DictTensor, B: int):
        raise NotImplementedError

    def update(self, info):
        raise NotImplementedError

    def __call__(
        self,
        state: DictTensor,
        input: DictTensor,
        agent_info: DictTensor,
        history: TemporalDictTensor = None,
    ):
        raise NotImplementedError

    def call_replay(self, trajectories: Trajectories, t: int, state):
        assert not self.require_history()
        info = trajectories.info
        if state is None:
            assert t == 0
            state = info.truncate_key("agent_state/")
        agent_info = info.truncate_key("agent_info/").to("cpu")
        tslice = trajectories.trajectories.temporal_index(t)
        observation = tslice.truncate_key("observation/")
        action, state = self.__call__(state, observation, agent_info, None)

        return action, state

    def close(self):
        pass

class RL_Agent_CheckDevice(RL_Agent):
    """This class is used to check that an Agent is working correctly on a particular device
    It does not modify the behaviour of the agent, but check that input/output are on the right devices
    """
    def __init__(self,agent,device):
        self.agent=agent
        self.device=device

    def require_history(self):
        return self.agent.require_history()

    def initial_state(self, agent_info: DictTensor, B: int):
        assert agent_info.empty() or agent_info.device()==torch.device("cpu"),"agent_info has to be on CPU"
        i=self.agent.initial_state(agent_info,B)
        assert i.empty() or i.device()==self.device,"[RL_CheckDeviceAgent] initial_state on wrong device"
        return i

    def update(self, info):
        self.agent.update(info)

    def __call__(
        self,
        state: DictTensor,
        input: DictTensor,
        agent_info: DictTensor,
        history: TemporalDictTensor = None,
    ):
        assert state.empty() or state.device()==self.device,"[RL_CheckDeviceAgent] state on wrong device"
        assert input.empty() or input.device()==self.device,"[RL_CheckDeviceAgent] input on wrong device"
        assert agent_info.empty() or agent_info.device()==torch.device("cpu"),"agent_info has to be on CPU"
        assert history is None or history.empty() or history.device()==self.device,"[RL_CheckDeviceAgent] history on wrong device"
        action,new_state=self.agent(state,input,agent_info,history)
        assert action.device()==self.device,"[RL_CheckDeviceAgent] action on wrong device"
        assert new_state.empty() or new_state.device()==self.device,"[RL_CheckDeviceAgent] new_state on wrong device"
        return action,new_state

    def call_replay(self, trajectories: Trajectories, t: int, state):
        print(trajectories.device())
        print(self.device)
        print("===")
        assert trajectories.device()==self.device,"[RL_CheckDeviceAgent] trajectories on wrong device"
        return self.agent.call_replay(trajectories,t,state)

    def close(self):
        self.agent.close()

def replay_agent_stateless(agent, trajectories: Trajectories, replay_method_name: str):
    """
    Replay transitions all in one
    returns a TDT
    """
    f = getattr(agent, replay_method_name)
    return f(trajectories)


def replay_agent(
    agent, trajectories: Trajectories, replay_method_name: str = "call_replay"
):
    """
    Replay transitions one by one in the temporal order, passing a state between each call
    returns a TDT
    """
    T = trajectories.trajectories.lengths.max().item()
    f = getattr(agent, replay_method_name)

    output, state = f(trajectories, 0, None)
    variables = {}

    for k in output.keys():
        s = output[k].size()
        t = torch.zeros(s[0], T, *s[1:], dtype=output[k].dtype).to(
            trajectories.device()
        )
        t[:, 0] = output[k]
        variables[k] = t

    for t in range(1, T):
        output, state = f(trajectories, t, state)
        for k in output.keys():
            variables[k][:, t] = output[k]
    tdt = TemporalDictTensor(
        variables, lengths=trajectories.trajectories.lengths.clone()
    )
    return tdt
