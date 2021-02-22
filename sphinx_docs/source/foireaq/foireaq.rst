FAQ
===

* *Q: Why would I use RLStructures ?*

Because it will speed-up your RL development process, particularly if you are starting a new project, and particularly if your objective is either to deal with a particular non-classical RL setting (unsupervised, multitask, lifelong learning, etc…), complex agents, or developing/testing new ideas. If you aim to run existing algorithms on particular environments then you will find better open source libraries where standard algorithms have been optimized better than our implementations. But if you want to modify or create algorithms, we think that RLStructures is a good choice.

Moreover, we will help you at any stage of your project, and we have a small group of users that can help you.

* *Q: Can I find more detailed documentation to really understand RLStructures?*

A: Multiple tutorials are provided to make the described principles concrete. More tutorials will be proposed in the next months.

See: https://medium.com/@ludovic.den

* *Q:  Is it complicated to use existing environments with RLStructures?*

A: RLStructures comes with OpenAI Gym wrappers allowing users to quickly adapt a classical environment to RLStructures in one line of code.

* *Q: You say that RLStructures allows the modelling of the interactions of multiple agents with multiple environments at once. What does this mean?*

A: Usually, RL platforms define an API for coding a single policy, or a single environment. In RLStructures, an Agent and an Environment are ‘by nature’ parametrized by multiple input tensors (a DictTensor). A policy thus corresponds to an Agent(z) where z is the parameters of the agent. It is the same for an environment which is parametrized by an input value w such that Environment(w) is a classical MDP. RLStructures allows one to gather, in parallel, N trajectories or episodes parametrized by agent parameters [z1,z2,...,zN] with environments parametrized by [w1,...,wN] such that this trajectory/episode acquisition actually corresponds to executing N different agents over N different environments.

As a simple example, one can for instance use the agent parameters z as the value of the epsilon in epsilon-greedy policies, such that trajectories with multiple values of epsilon can be sampled in parallel.

This is a very powerful mechanism for practical problems e.g multitask learning where the environment parameters correspond to a task to solve, mixtures of policies where the agent parameters are the identifier of the expert to use, etc.

* *Q: You say that RLStructures allows one to implement complex policies. Can you tell me more?*

A: The Agent API we provide is simple but powerful. It expects that an agent has an internal state (as a DictTensor which is a dictionary of tensors), and produces actions (as a DictTensor) and a next state (also as a DictTensor). The internal state can thus be very complex, containing for instance the state of multiple modules in the policy (e.g for hierarchical policies). And the action output can also be very complex, for instance containing actions computed at multiple levels in a hierarchical policy. When collecting trajectories, all these values will be made available to the user such that complex loss functions will be easy to compute, even if the mechanism of the agent may be complex.

Moreover, an Agent can typically use multiple Pytorch Models (e.g at each level of the hierarchy) and there are few limits to what an Agent can do.

We provide such examples of policies in the tutorial directory, and we are progressively extending this catalog of policies.

* *Q: You talk about Transformer-based policies...*

A: When using particular batchers, an agent has access to the complete trajectory history, such that it can compute actions through neural networks architectures that use the whole history, like transformer networks. In RLStructures, it is completely feasible. Moreover, when facing too long trajectories it is possible to constraint the agent to access only the n last timesteps.

* *Q: If I execute 128 agents over 128 environments, it will take time…. Do I have to wait until it is finished?*

No, RLStructures allows the use of batchers in blocking and non-blocking modes such that you can typically launch the acquisition process, do other computations, and get back the acquired trajectories when the acquisition is finished, without wasting time.

* *Q: Can RLStructures be used asynchronously e.g by having processes infinitely acquiring information (like in IMPALA)?*

Yes, instead of asking a batcher to return the computed trajectories, the user can directly address the memory buffer where the trajectories are stored, such that the batcher will work at maximum speed. It will be illustrated at the release of the IMPALA algorithm in the repository.

* *Q: Can I use multiple CPUs and GPUs?*

Yes, a typical RL implementation is to have the batchers working on multiple CPUs, and loss computation on one or multiple GPUs (since the loss computation may be done on large batches of trajectories). This is the schema we propose in our examples. But RLStructures also allows one to have batchers working on GPUs (we have already done this in some experiments).

* *Q: What is the speed of RLStructures?*

RLStructures is certainly not the fastest RL library since it is based on a general setting where an agent is executed on environments to generate trajectories (applying some ‘forward’ calls). Then (see examples implementations) the loss computation is usually made in a second step over bigger batches of trajectories, needing to re-execute a forward pass (which can be faster, for instance by using GPUs and recomputing only some variables) to compute the resulting gradient. Some other libraries allow one to keep the ‘batcher forward pass’ in memory, avoiding this double computation, but at the price of less readability and less expressivity. Note also that we do not spend as much energy to optimize RLStructures speed during development. Anyway, when comparing to a ‘naive’ implementation, and by allowing an easy use of multiple processes, RLStructures can largely speed-up any implementation.

* *Q: Can I use Tensorflow or Jax with RLStructures ?*

No way!! DictTensor and TemporalDictTensor are extensions of pytorch tensors, and RLStructures is heavily dependent on PyTorch.
