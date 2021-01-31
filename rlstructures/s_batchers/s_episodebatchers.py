#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from rlstructures import TemporalDictTensor, DictTensor
from .s_buffers import S_Buffer
from .s_threadworker import S_ThreadWorker
import rlstructures.logging as logging
import torch
import numpy as np
import copy

class S_EpisodeBatcher:
    def execute(self, n_episodes, agent_info=None,env_info=None):
        n_workers = len(self.workers)
        assert n_episodes % (self.n_envs*n_workers) == 0

        self.n_per_worker = [int(n_episodes / n_workers) for w in range(n_workers)]
        pos=0
        for k in range(n_workers):
                wi,ei=None,None
                n=self.n_per_worker[k]
                assert n%self.n_envs==0

                if not agent_info is None:
                    wi=agent_info.slice(pos,pos+n)
                if not env_info is None:
                    ei=env_info.slice(pos,pos+n)
                self.workers[k].acquire_episodes(
                    n_episodes=self.n_per_worker[k], agent_info=wi, env_info=ei
                )
                pos+=n
        assert pos==n_episodes

    def reexecute(self):
        n_workers = len(self.workers)

        for k in range(n_workers):
                self.workers[k].acquire_episodes_again()

    def get(self,blocking=True):
        if not blocking:
            for w in range(len(self.workers)):
                b=self.workers[w].finished()
                if not b:
                    return None,None

        max_length = 0
        buffer_slot_id_lists = []
        for w in range(len(self.workers)):
            if self.n_per_worker[w] > 0:
                sid_lists = self.workers[w].get()
                for sids in sid_lists:
                    buffer_slot_id_lists.append(sids)
                    max_length = max(max_length, len(sids))
        if max_length > 1:
            assert False,"S_EpisodeBatcher: episodes are on multiple slots..."
        else:
            buffer_slot_ids = [i[0] for i in buffer_slot_id_lists]
            return self.buffer.get_single_slots(buffer_slot_ids, erase=True)

    def update(self, info):
        for w in self.workers:
            w.update_worker(info)

    def close(self):
        for w in self.workers:
            w.close()

    def __init__(
        self,
        n_timesteps,
        n_slots,
        create_agent,
        agent_args,
        create_env,
        env_args,

        n_threads,
        seeds=None,
    ):
        # Buffer creation:
        agent = create_agent(**agent_args)
        env = create_env(**{**env_args,"seed":0})

        obs,who=env.reset()
        B=obs.n_elems()
        with torch.no_grad():
            b,a=agent(None,obs,agent.get_default_agent_info(B))

        self.n_envs=env.n_envs()
        specs_agent_state=a.specs()
        specs_agent_output=b.specs()
        specs_environment=obs.specs()
        specs_agent_info=agent.get_default_agent_info(B).specs()
        specs_env_info=env.get_default_env_info().specs()
        specs_agent_info=agent.get_default_agent_info(1).specs()
        specs_env_info=env.get_default_env_info().specs()
        del env
        del agent

        self.buffer = S_Buffer(
            n_slots=n_slots,
            s_slots=n_timesteps,
            specs_agent_state=specs_agent_state,
            specs_agent_output=specs_agent_output,
            specs_environment=specs_environment,
            specs_agent_info=specs_agent_info,
            specs_env_info=specs_env_info
        )
        self.workers = []
        self.n_per_worker = []
        self.warning = False

        if seeds is None:
            logging.info(
                "Seeds for batcher environments has not been chosen. Default"
                + " is None"
            )
            seeds = [None for k in range(n_threads)]

        if (isinstance(seeds,int)):
            s=seeds
            seeds=[s+k*64 for k in range(n_threads)]
        assert len(seeds)==n_threads,"You have to choose one seed per thread"
        logging.info("[S_EpisodeBatcher] Creating %d threads" % (n_threads))
        for k in range(n_threads):
            e_args = {**env_args, "seed": seeds[k]}
            worker = S_ThreadWorker(
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
