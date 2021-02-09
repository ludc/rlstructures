#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import TemporalDictTensor,  DictTensor, Trajectories
from .tools import S_Buffer
from .tools import S_ProcessWorker
import torch
import numpy as np


class RL_Batcher:
    def reset(self,agent_info=DictTensor({}), env_info=DictTensor({})):
        n_workers = len(self.workers)
        pos=0
        for k in range(n_workers):
                n=self.n_envs
                wi=agent_info.slice(pos,pos+n)
                ei=env_info.slice(pos,pos+n)
                self.workers[k].reset(
                    agent_info=wi, env_info=ei
                )
                pos+=n
        assert agent_info.empty() or agent_info.n_elems()==pos
        assert env_info.empty() or env_info.n_elems()==pos

    def execute(self,agent_info=None):
        n_workers = len(self.workers)
        pos=0
        for k in range(n_workers):
                n=self.n_envs
                wi=None
                if not agent_info is None:
                    wi=agent_info.slice(pos,pos+n)
                self.workers[k].acquire_slot(wi)
                pos+=n



    def get(self,blocking=True):
        if not blocking:
            for w in range(len(self.workers)):
                if not self.workers[w].finished():
                    return None,None

        buffer_slot_ids = []
        n_still_running=0
        for w in range(len(self.workers)):
            bs,n = self.workers[w].get()
            buffer_slot_ids +=bs
            n_still_running+=n
        if len(buffer_slot_ids)==0:
            assert False,"Don't call batcher.get when all environnments are finished"

        slots,info = self.buffer.get_single_slots(buffer_slot_ids, erase=True)
        assert not slots.lengths.eq(0).any()
        return Trajectories(info,slots),n_still_running

    def update(self, info):
        for w in self.workers:
            w.update_worker(info)

    def close(self):
        for w in self.workers:
            w.close()
        for w in self.workers:
            del w

    def n_elems(self):
        return self._n_episodes

    def __init__(
        self,
        n_timesteps,
        create_agent,
        agent_args,
        create_env,
        env_args,
        n_processes,
        seeds,
        agent_info,
        env_info
    ):
# Buffer creation:
        agent = create_agent(**agent_args)
        env = create_env(**{**env_args,"seed":0})
        if not agent_info.empty():
            agent_info=agent_info.slice(0,1)
            agent_info=DictTensor.cat([agent_info for k in range(env.n_envs())])
        if not env_info.empty():
            env_info=env_info.slice(0,1)
            env_info=DictTensor.cat([env_info for k in range(env.n_envs())])

        obs,who=env.reset(env_info)
        B=obs.n_elems()
        with torch.no_grad():
            istate=agent.initial_state(agent_info,B)
            b,a=agent(istate,obs,agent_info)

        self.n_envs=env.n_envs()
        self._n_episodes=n_processes*self.n_envs

        specs_agent_state=a.specs()
        specs_agent_output=b.specs()
        specs_environment=obs.specs()
        specs_agent_info=agent_info.specs()
        specs_env_info=env_info.specs()
        del env
        del agent
        self.buffer = S_Buffer(
            n_slots=self.n_envs*n_processes,
            s_slots=n_timesteps,
            specs_agent_state=specs_agent_state,
            specs_agent_output=specs_agent_output,
            specs_environment=specs_environment,
            specs_agent_info=specs_agent_info,
            specs_env_info=specs_env_info
        )
        self.workers = []
        self.n_per_worker = []

        assert isinstance(seeds,list),"You have to choose one seed per process"
        assert len(seeds)==n_processes,"You have to choose one seed per process"

        print("[Batcher] Creating %d processes " % (n_processes))
        for k in range(n_processes):
            e_args = {**env_args, "seed": seeds[k]}
            worker = S_ProcessWorker(
                len(self.workers),
                create_agent,
                agent_args,
                create_env,
                e_args,
                self.buffer,
            )
            self.workers.append(worker)

    def close(self):
        super().close()
        self.buffer.close()
