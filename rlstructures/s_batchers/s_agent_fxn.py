#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from rlstructures import DictTensor

def s_acquire_slot(
    buffer,
    env,
    agent,
    agent_state,
    observation,
    agent_info,
    env_info,
    env_running,
):
    with torch.no_grad():
        B = env_running.size()[0]
        id_slots = buffer.get_free_slots(B)
        env_to_slot = {env_running[i].item(): id_slots[i] for i in range(len(id_slots))}

        to_write = (
                agent_info.prepend_key("agent_info/")
                +env_info.prepend_key("env_info/")
        )
        buffer.fwrite(id_slots, to_write)
        require_history=agent.require_history()

        t = 0
        for t in range(buffer.s_slots):
            # print(t,buffer.s_slots)
            _id_slots = [
                env_to_slot[env_running[i].item()] for i in range(env_running.size()[0])
            ]
            history=None
            if require_history:
                history=buffer.get_single_slots(_id_slots,erase=False)
            agent_output, new_agent_state = agent(
                agent_state, observation, agent_info, history=history
            )

            # print(old_agent_state,agent_output,new_agent_state)
            (nobservation, env_running), (nnobservation, nenv_running) = env.step(
                agent_output
            )
            position_in_slot=torch.tensor([t]).repeat(len(_id_slots))

            to_write = (
                observation.prepend_key("observation/")
                + agent_output.prepend_key("action/")
                + new_agent_state.prepend_key("_agent_state/")
                + nobservation.prepend_key("_observation/")
            )
            if not agent_state is None:
                to_write = to_write + agent_state.prepend_key("agent_state/")

            id_slots = [
                env_to_slot[env_running[i].item()] for i in range(env_running.size()[0])
            ]
            assert id_slots==_id_slots
            buffer.write(id_slots, to_write)

            # Now, let us prepare the next step

            observation = nnobservation
            idxs = [
                k
                for k in range(env_running.size()[0])
                if env_running[k].item() in nenv_running
            ]
            if len(idxs) == 0:
                return env_to_slot, None, None, None, None, nenv_running
            idxs = torch.tensor(idxs)

            agent_state = new_agent_state.index(idxs)
            agent_info = agent_info.index(idxs)
            env_running = nenv_running
            assert len(agent_state.keys()) == 0 or (
                agent_state.n_elems() == observation.n_elems()
            )

            if nenv_running.size()[0] == 0:
                return env_to_slot, agent_state, observation,  agent_info, env_info,env_running
        return env_to_slot, agent_state, observation, agent_info, env_info,env_running

def s_acquire_episodes(buffer,
    env,
    agent,
    env_info,
    agent_info,
):
    flag = True
    slots = None
    if env_info is None:
        env_info=env.get_default_env_info()

    observation, env_running = env.reset(env_info)
    B=observation.n_elems()
    slots = [[] for k in range(env.n_envs())]
    t = 0
    if agent_info is None:
        agent_info=agent.get_default_agent_info(B)
    agent_state=None
    assert not agent_info is None
    assert not env_info is None

    while True:
        env_to_slots, agent_state, observation, agent_info, env_info, env_running = s_acquire_slot(
            buffer, env, agent, agent_state, observation, agent_info, env_info,env_running
        )
        [slots[k].append(env_to_slots[k]) for k in env_to_slots]
        if env_running.size()[0] == 0:
            return tuple(slots)
        t = t + 1
