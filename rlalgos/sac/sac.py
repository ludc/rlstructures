#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from rlstructures.logger import Logger, TFLogger
from rlstructures import DictTensor, TemporalDictTensor
from rlstructures.batchers import Batcher, EpisodeBatcher
from rlstructures import RL_Batcher
from rlstructures.batchers.buffers import LocalBuffer
from rlstructures import logging
from rlstructures.tools import weight_init
import torch.nn as nn
import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
import pickle


class ReplayBuffer:
    """
    This class is used to store transitions. Each transition is a TemporalDictTensor of size T
    """

    def __init__(self, N):
        self.N = N
        self.buffer = None
        self.info_buffer = None

    def _init_buffer(self, trajectories):
        self.buffer = {}
        for k in trajectories.keys():
            dtype = trajectories[k].dtype
            size = trajectories[k].size()
            b_size = (self.N,) + size[2:]
            self.buffer[k] = torch.zeros(*b_size, dtype=dtype)

        self.pos = 0
        self.full = False

    def write(self, trajectories):
        rs = {}
        new_pos = None
        for k in trajectories.keys():
            v = trajectories[k]
            size = v.size()
            b_size = (size[0] * size[1],) + size[2:]
            v = v.reshape(*b_size)
            n = v.size()[0]
            overhead = self.N - (self.pos + n)
            if new_pos is None:
                new_pos = torch.arange(n) + self.pos
                mask = new_pos.ge(self.N).float()
                nidx = torch.arange(n) + self.pos - self.N
                new_pos = (new_pos * (1 - mask) + mask * (nidx)).long()

            self.buffer[k][new_pos] = v

        self.pos = self.pos + n
        if self.pos >= self.N:
            self.pos = self.pos - self.N
            self.full = True
        assert self.pos < self.N

    def size(self):
        if self.full:
            return self.N
        else:
            return self.pos

    def push(self, trajectories):
        """
        Add transitions to the replay buffer
        """
        trajectories = trajectories.trajectories
        max_length = trajectories.lengths.max().item()
        assert trajectories.lengths.eq(max_length).all()
        if self.buffer is None:
            self._init_buffer(trajectories)
        self.write(trajectories)

    def sample(self, n=1):
        limit = self.pos
        if self.full:
            limit = self.N
        transitions = torch.randint(0, high=limit, size=(n,))
        d = {k: self.buffer[k][transitions] for k in self.buffer}
        return DictTensor(d)


class SAC:
    def __init__(self, config, create_train_env, create_env, create_agent):
        self.config = config

        # Creation of the Logger (that saves in tensorboard and CSV)
        self.logger = TFLogger(log_dir=self.config["logdir"], hps=self.config)

        self._create_env = create_env
        self._create_train_env = create_train_env
        self._create_agent = create_agent

        # Creation of one env instance to get the dimensionnality of observations and number of actions
        env = self._create_env(
            self.config["n_envs"], seed=0, env_name=self.config["env_name"]
        )
        self.action_dim = env.action_space.sample().shape[0]
        self.obs_dim = env.reset()[0]["frame"].size()[1]
        del env

    def _state_dict(self, model, device):
        sd = model.state_dict()
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def run(self):
        self.q1 = self._create_q()
        self.q2 = self._create_q()
        self.target_q1 = self._create_q()
        self.target_q2 = self._create_q()
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.learning_model = self._create_model()

        model = copy.deepcopy(self.learning_model)
        self.evaluation_batcher = RL_Batcher(
            n_timesteps=self.config["max_episode_steps"],
            create_agent=self._create_agent,
            create_env=self._create_env,
            env_args={
                "n_envs": self.config["n_evaluation_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
            },
            agent_args={"action_dim": self.action_dim, "policy": model},
            n_processes=self.config["n_evaluation_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_evaluation_processes"])
            ],
            agent_info=DictTensor({"stochastic": torch.tensor([True])}),
            env_info=DictTensor({}),
        )

        model = copy.deepcopy(self.learning_model)
        self.train_batcher = RL_Batcher(
            n_timesteps=self.config["batch_timesteps"],
            create_agent=self._create_agent,
            create_env=self._create_train_env,
            env_args={
                "n_envs": self.config["n_envs"],
                "max_episode_steps": self.config["max_episode_steps"],
                "env_name": self.config["env_name"],
            },
            agent_args={"action_dim": self.action_dim, "policy": model},
            n_processes=self.config["n_processes"],
            seeds=[
                self.config["env_seed"] + k * 10
                for k in range(self.config["n_processes"])
            ],
            agent_info=DictTensor({"stochastic": torch.tensor([True])}),
            env_info=DictTensor({}),
        )

        self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"])
        device = torch.device(self.config["learner_device"])
        self.learning_model.to(device)

        self.q1.to(device)
        self.q2.to(device)
        self.target_q1.to(device)
        self.target_q2.to(device)
        optimizer = torch.optim.Adam(
            self.learning_model.parameters(), lr=self.config["lr"]
        )
        optimizer_q1 = torch.optim.Adam(self.q1.parameters(), lr=self.config["lr"])
        optimizer_q2 = torch.optim.Adam(self.q2.parameters(), lr=self.config["lr"])

        self.train_batcher.update(
            self._state_dict(self.learning_model, torch.device("cpu"))
        )
        self.evaluation_batcher.update(
            self._state_dict(self.learning_model, torch.device("cpu"))
        )

        n_episodes = self.config["n_envs"] * self.config["n_processes"]
        self.train_batcher.reset(
            agent_info=DictTensor({"stochastic": torch.zeros(n_episodes).eq(0.0)})
        )
        logging.info("Sampling initial transitions")
        n_iterations = int(
            self.config["n_starting_transitions"]
            / (n_episodes * self.config["batch_timesteps"])
        )
        for k in range(n_iterations):
            self.train_batcher.execute()
            trajectories, n = self.train_batcher.get()
            self.replay_buffer.push(trajectories)
        print("replay_buffer_size = ", self.replay_buffer.size())

        n_episodes = (
            self.config["n_evaluation_envs"] * self.config["n_evaluation_processes"]
        )
        stochastic = torch.tensor(
            [self.config["evaluation_mode"] == "stochastic"]
        ).repeat(n_episodes)
        self.evaluation_batcher.reset(agent_info=DictTensor({"stochastic": stochastic}))
        self.evaluation_batcher.execute()
        self.evaluation_iteration = 0

        logging.info("Starting Learning")
        _start_time = time.time()

        logging.info("Learning")
        self.iteration = 0
        while time.time() - _start_time < self.config["time_limit"]:
            self.train_batcher.execute()
            trajectories, n = self.train_batcher.get()
            self.replay_buffer.push(trajectories)
            self.logger.add_scalar(
                "replay_buffer_size", self.replay_buffer.size(), self.iteration
            )
            # avg_reward = 0

            for k in range(self.config["n_batches_per_epochs"]):
                transitions = self.replay_buffer.sample(n=self.config["size_batches"])

                # print(dt)
                dt, transitions = self.get_q_loss(transitions, device)
                [
                    self.logger.add_scalar(k, dt[k].item(), self.iteration)
                    for k in dt.keys()
                ]
                optimizer_q1.zero_grad()
                dt["q1_loss"].backward()
                optimizer_q1.step()

                optimizer_q2.zero_grad()
                dt["q2_loss"].backward()
                optimizer_q2.step()

                optimizer.zero_grad()
                dt = self.get_policy_loss(transitions)
                [
                    self.logger.add_scalar(k, dt[k].item(), self.iteration)
                    for k in dt.keys()
                ]
                dt["policy_loss"].backward()
                optimizer.step()

                tau = self.config["tau"]
                self.soft_update_params(self.q1, self.target_q1, tau)
                self.soft_update_params(self.q2, self.target_q2, tau)

                self.iteration += 1

            self.train_batcher.update(
                self._state_dict(self.learning_model, torch.device("cpu"))
            )

            evaluation_trajectories, n = self.evaluation_batcher.get(blocking=False)
            if not evaluation_trajectories is None:  # trajectories are available
                # Compute the cumulated reward
                cumulated_reward = (
                    (
                        evaluation_trajectories.trajectories["_observation/reward"]
                        * evaluation_trajectories.trajectories.mask()
                    )
                    .sum(1)
                    .mean()
                )
                self.logger.add_scalar(
                    "evaluation_reward",
                    cumulated_reward.item(),
                    self.evaluation_iteration,
                )
                print(
                    "At iteration %d, reward is %f"
                    % (self.evaluation_iteration, cumulated_reward.item())
                )
                # We reexecute the evaluation batcher (with same value of agent_info and same number of episodes)
                self.evaluation_batcher.update(self.learning_model.state_dict())
                self.evaluation_iteration = self.iteration
                n_episodes = (
                    self.config["n_evaluation_processes"]
                    * self.config["n_evaluation_envs"]
                )
                agent_info = DictTensor(
                    {"stochastic": torch.tensor([False]).repeat(n_episodes)}
                )
                self.evaluation_batcher.reset(agent_info=agent_info)
                self.evaluation_batcher.execute()

    def get_q_loss(self, transitions, device):

        transitions = transitions.to(device)
        B = transitions.n_elems()
        Bv = torch.arange(B)
        action = transitions["action/action"]
        reward = transitions["_observation/reward"]
        frame = transitions["observation/frame"]
        _frame = transitions["_observation/frame"]
        _done = transitions["_observation/done"].float()

        # action for s_prime
        mean_prime, var_prime = self.learning_model(_frame)
        _id = torch.eye(self.action_dim).unsqueeze(0).repeat(B, 1, 1)
        # _nvar = var_prime.unsqueeze(-1).repeat(1, 1, self.action_dim)
        # _nvar = _nvar * _id
        distribution = torch.distributions.Normal(mean_prime, var_prime)
        next_action = distribution.sample().detach()

        # Compute targets
        q1 = self.target_q1(_frame, next_action).detach().squeeze(-1)
        q2 = self.target_q2(_frame, next_action).detach().squeeze(-1)
        q = torch.min(q1, q2)
        lp = distribution.log_prob(next_action).detach().sum(-1)
        q = q - self.config["lambda_entropy"] * lp
        target_value = q * (1.0 - _done) * self.config["discount_factor"] + reward

        q1_loss = (target_value.detach() - self.q1(frame, action).squeeze(-1)) ** 2
        q2_loss = (target_value.detach() - self.q2(frame, action).squeeze(-1)) ** 2
        dt = {
            "q1_loss": q1_loss.mean(),
            "q2_loss": q2_loss.mean(),
        }
        return DictTensor(dt), transitions

    def get_policy_loss(self, transitions):
        frame = transitions["observation/frame"]
        B = transitions.n_elems()
        # Now, compute the policy term
        mean, var = self.learning_model(frame)
        # print(var.mean().item())
        # print(mean)
        _id = torch.eye(self.action_dim).unsqueeze(0).repeat(B, 1, 1)
        # _nvar = var.unsqueeze(-1).repeat(1, 1, self.action_dim)
        # _nvar = _nvar * _id
        distribution = torch.distributions.Normal(mean, var)
        entropy = distribution.entropy().mean()
        action_tilde = distribution.rsample()
        # print(action_tilde)
        q1 = self.q1(frame, action_tilde).squeeze(-1)
        q2 = self.q2(frame, action_tilde).squeeze(-1)
        q = torch.min(q1, q2)
        loss = q - self.config["lambda_entropy"] * distribution.log_prob(
            action_tilde
        ).sum(-1)

        dt = {
            "policy_loss": -loss.mean(),
            "entropy": entropy.detach(),
            "avg_var": var.mean().detach(),
            "avg_mean": mean.mean().detach(),
        }
        dt = DictTensor(dt)
        return dt
