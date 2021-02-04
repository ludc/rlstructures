#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.multiprocessing as mp
import time
import math

import torch
import torch.multiprocessing as mp
from rlstructures import TemporalDictTensor,DictTensor

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
        if agent_state is None:
            agent_state=agent.initial_state(agent_info,B)

        id_slots = buffer.get_free_slots(B)
        env_to_slot = {env_running[i].item(): id_slots[i] for i in range(len(id_slots))}
        to_write = (
                agent_info.prepend_key("agent_info/")
                +env_info.prepend_key("env_info/")
                +agent_state.prepend_key("agent_state/")
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
                + nobservation.prepend_key("_observation/")
            )

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
            env_info = env_info.index(idxs)
            env_running = nenv_running
            assert len(agent_state.keys()) == 0 or (
                agent_state.n_elems() == observation.n_elems()
            )

            if nenv_running.size()[0] == 0:
                return env_to_slot, agent_state, observation,  agent_info, env_info,env_running
        return env_to_slot, agent_state, observation, agent_info, env_info,env_running


class S_Buffer:
    """
    Defines a shared buffer to store trajectories / transitions
    The buffer is structured as nslots of size s_slots for each possible variable
    """

    def __init__(
        self,
        n_slots=None,
        s_slots=None,
        specs_agent_state=None,
        specs_agent_output=None,
        specs_environment=None,
        specs_agent_info=None,
        specs_env_info=None,
        device=torch.device("cpu"),
    ):
        """
        Init a new buffer

        Args:
            n_slots (int): the number of slots
            s_slots (int): the size of each slot (temporal dimension)
            specs (dict): The description of the variable to store in the buffer
        """
        self._device = device
        self.buffers = {}
        self.fbuffers = {}

        self.n_slots = n_slots
        self.s_slots = s_slots

        # Creation of the storage buffers
        nspecs_env = {"_observation/" + k: specs_environment[k] for k in specs_environment}
        specs_environment = {"observation/" + k: specs_environment[k] for k in specs_environment}
        specs_agent_output = {"action/" + k: specs_agent_output[k] for k in specs_agent_output}

        specs = {
            **specs_agent_output,
            **specs_environment,
            **nspecs_env,
        }

        for n in specs:
            size = (n_slots, s_slots) + specs[n]["size"]
            print(
                "Creating buffer for '"
                + n
                + "' of size "
                + str(size)
                + " and type "
                + str(specs[n]["dtype"])
            )
            assert not n in self.buffers, "Same key is used by the agent and the env"
            self.buffers[n] = (
                torch.zeros(size, dtype=specs[n]["dtype"])
                .to(self._device)
                .share_memory_()
            )

        specs_agent_info = {"agent_info/" + k: specs_agent_info[k] for k in specs_agent_info}
        specs_env_info = {"env_info/" + k: specs_env_info[k] for k in specs_env_info}
        specs_agent_state = {"agent_state/"+k:specs_agent_state[k] for k in specs_agent_state}
        specs_info = {**specs_agent_info,**specs_env_info,**specs_agent_state}
        for n in specs_info:
            size = (n_slots, ) + specs_info[n]["size"]
            print(
                "Creating F buffer for '"
                + n
                + "' of size "
                + str(size)
                + " and type "
                + str(specs_info[n]["dtype"])
            )
            assert not n in self.fbuffers, "Same key is used by the agent and the env"
            self.fbuffers[n] = (
                torch.zeros(size, dtype=specs_info[n]["dtype"])
                .to(self._device)
                .share_memory_()
            )

        self.position_in_slot = (
            torch.zeros(n_slots).to(self._device).long().share_memory_()
        )
        self._free_slots_queue = mp.Queue()
        self._free_slots_queue.cancel_join_thread()
        for i in range(n_slots):
            self._free_slots_queue.put(i, block=True)
        self._full_slots_queue = mp.Queue()
        self._full_slots_queue.cancel_join_thread()

    def device(self):
        return self._device

    def get_free_slots(self, k):
        """
        Returns k available slots. Wait until enough slots are free
        """
        assert k > 0
        x = [self._free_slots_queue.get() for i in range(k)]
        for i in x:
            self.position_in_slot[i] = 0
        return x

    def set_free_slots(self, s):
        """
        Tells the buffer that it can reuse the given slots
        :param s may be one slot (int) or multiple slots (list of int)
        """
        assert not s is None
        if isinstance(s, int):
            self._free_slots_queue.put(s)
        else:
            for ss in s:
                self._free_slots_queue.put(ss)

    def fwrite(self,slots,variables):
        if variables.empty():
            return

        if not variables.device() == self._device:
            variables = variables.to(self._device)

        slots = torch.tensor(slots).to(self._device)
        assert variables.n_elems() == len(slots)
        a = torch.arange(len(slots)).to(self._device)
        # print("Write in "+str(slot)+" at positions "+str(position))
        for n in variables.keys():
            #print("FWrite ",n,":", variables[n]," in ",slots)
            # assert variables[n].size()[0] == 1
            # print(self.buffers[n][slots].size())
            self.fbuffers[n][slots] = variables[n][a].detach()

    def write(self, slots, variables):
        if variables.empty():
            return

        if not variables.device() == self._device:
            variables = variables.to(self._device)

        slots = torch.tensor(slots).to(self._device)
        assert variables.n_elems() == len(slots)
        positions = self.position_in_slot[slots]
        a = torch.arange(len(slots)).to(self._device)
        # print("Write in "+str(slot)+" at positions "+str(position))
        for n in variables.keys():
            #print("Write ",n,":", variables[n]," in ",slots)
            # assert variables[n].size()[0] == 1
            # print(self.buffers[n][slots].size())
            self.buffers[n][slots, positions] = variables[n][a].detach()
        self.position_in_slot[slots] += 1

    def is_slot_full(self, slot):
        """
        Returns True of a slot is full
        """
        return self.position_in_slot[slot] == self.s_slots

    def get_single(self,slots,position):
        assert isinstance(slots, list)
        assert isinstance(slots[0], int)
        idx = torch.tensor(slots).to(self._device).long()
        d={k:self.buffers[k][idx,position] for k in self.buffers}
        return DictTensor(d)

    def close(self):
        """
        Close the buffer
        """
        self._free_slots_queue.close()
        self._full_slots_queue.close()

    def get_single_slots(self, slots, erase=True, clone=True):
        assert isinstance(slots, list)
        assert isinstance(slots[0], int)
        idx = torch.tensor(slots).to(self._device).long()
        lengths = self.position_in_slot[idx]
        ml = lengths.max().item()
        if not clone:
            v = {k: self.buffers[k][idx, :ml] for k in self.buffers}
            fvalues=DictTensor({k:self.fbuffers[k][idx] for k in self.fbuffers})
        else:
            v = {k: self.buffers[k][idx, :ml].clone() for k in self.buffers}
            fvalues=DictTensor({k:self.fbuffers[k][idx].clone() for k in self.fbuffers})
        if erase:
            self.set_free_slots(slots)
        tdt = TemporalDictTensor(v, lengths)
        return (tdt,fvalues)


def s_worker_process(
    buffer,
    create_env,
    env_parameters,
    create_agent,
    agent_parameters,
    in_queue,
    out_queue,
):
    env = create_env(**env_parameters)
    n_envs = env.n_envs()
    agent = create_agent(**agent_parameters)

    agent_state = None
    observation = None
    env_running = None
    agent_info = None
    env_info = None
    n_episodes = None
    terminate_process = False
    while not terminate_process:
        order = in_queue.get()
        assert isinstance(order, tuple)
        order_name = order[0]
        if order_name == "close":
            logging.debug("\tClosing process...")
            terminate_process = True
            env.close()
            agent.close()
        elif order_name == "reset":
            _, _agent_info, _env_info = order
            agent_info=_agent_info.clone()
            env_info=_env_info.clone()
            del(_agent_info)
            del(_env_info)
            agent_state = None
            observation = None
            env_running = None
            assert agent_info.empty() or agent_info.n_elems()==env.n_envs()
            assert env_info.empty() or env_info.n_elems()==env.n_envs()
            observation, env_running = env.reset(env_info)
        elif order_name == "slot":
            if len(env_running)==0:
                out_queue.put([])
            else:
                if not order[1] is None:
                    agent_info=order[1]
                    assert agent_info.n_elems()==len(env_running)
                env_to_slot, agent_state, observation, agent_info,env_info, env_running = s_acquire_slot(
                        buffer,
                        env,
                        agent,
                        agent_state,
                        observation,
                        agent_info,
                        env_info,
                        env_running,
                )
                slots=[env_to_slot[k] for k in env_to_slot]
                out_queue.put((slots,len(env_running)))
        elif order_name == "update":
            agent.update(order[1])
            out_queue.put("ok")
        else:
            assert False, "Unknown order..."
    out_queue.put("TERMINATED")


class S_ProcessWorker:
    def __init__(
        self, worker_id, create_agent, agent_args, create_env, env_args, buffer
    ):
        self.worker_id = worker_id
        ctx = mp.get_context("spawn")
        self.inq = ctx.Queue()
        self.outq = ctx.Queue()
        self.inq.cancel_join_thread()
        self.outq.cancel_join_thread()
        p = ctx.Process(
            target=s_worker_process,
            args=(
                buffer,
                create_env,
                env_args,
                create_agent,
                agent_args,
                self.inq,
                self.outq,
            ),
        )
        self.process = p
        p.daemon = True
        p.start()

    def acquire_slot(self,agent_info=None):
        order = ("slot", agent_info)
        self.inq.put(order)

    def acquire_state(self):
        order = ("acquire_state")
        self.inq.put(order)

    def get_state(self):
        t=self.outq.get()
        return t

    def reset(self,agent_info=None, env_info=None):
        order = ("reset", agent_info, env_info)
        self.inq.put(order)

    def finished(self):
        try:
            r=self.outq.get(False)
            self.outq.put(r)
            return True
        except:
            return False

    def get(self):
        t=self.outq.get()
        return t

    def update_worker(self, info):
        self.inq.put(("update", info))
        self.outq.get()

    def close(self):
        logging.debug("Stop process " + str(self.worker_id))
        self.inq.put(("close",))
        self.outq.get()
        time.sleep(0.1)
        self.process.terminate()
        self.process.join()
        self.inq.close()
        self.outq.close()
        time.sleep(0.1)
        del self.inq
        del self.outq
