from rlstructures.env import VecEnv
from rlstructures import DictTensor
import torch

class DeviceEnv:
    def __init__(self,env,device):
        self.env=env
        self.device=device
        self.action_space=self.env.action_space

    def reset(self, env_info=DictTensor({})):
        assert env_info.empty() or env_info.device()==torch.device("cpu"),"env_info must be on CPU"
        o,e=self.env.reset(env_info)
        return o.to(self.device),e.to(self.device)

    def step(self, policy_output):
        assert policy_output.device()==self.device
        (a,b),(c,d)=self.env.step(policy_output)
        return (a.to(self.device),b.to(self.device)),(c.to(self.device),d.to(self.device))

    def close(self):
        self.env.close()

    def n_envs(self):
       return self.env.n_envs()
