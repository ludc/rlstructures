#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.multiprocessing as mp
import rlstructures.logging as logging
from rlstructures import TemporalDictTensor,DictTensor


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
            logging.getLogger("buffer").debug(
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
            logging.getLogger("buffer").debug(
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
        # logging.getLogger("buffer").debug("GET FREE " + str(x))
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
        # logging.getLogger("buffer").debug("SET FREE " + str(s))

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

    def get_single_slots(self, slots, erase=True):
        assert isinstance(slots, list)
        assert isinstance(slots[0], int)
        idx = torch.tensor(slots).to(self._device).long()
        lengths = self.position_in_slot[idx]
        ml = lengths.max().item()
        v = {k: self.buffers[k][idx, :ml] for k in self.buffers}
        fvalues=DictTensor({k:self.fbuffers[k][idx] for k in self.fbuffers})
        if erase:
            self.set_free_slots(slots)
        tdt = TemporalDictTensor(v, lengths)
        return (tdt,fvalues)
