#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from .s_agent_fxn import s_acquire_slot
import torch.multiprocessing as mp
import rlstructures.logging as logging
import time
import math

def s_worker_thread(
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
    terminate_thread = False
    while not terminate_thread:
        order = in_queue.get()
        assert isinstance(order, tuple)
        order_name = order[0]
        if order_name == "close":
            logging.debug("\tClosing thread...")
            terminate_thread = True
            env.close()
            agent.close()
        elif order_name == "reset":
            _, agent_info, env_info = order
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


class S_ThreadWorker:
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
            target=s_worker_thread,
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

    def acquire_slot(self):
        order = ("slot", None)
        self.inq.put(order)

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
        return self.outq.get()

    def update_worker(self, info):
        self.inq.put(("update", info))
        self.outq.get()

    def close(self):
        logging.debug("Stop Thread " + str(self.worker_id))
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
