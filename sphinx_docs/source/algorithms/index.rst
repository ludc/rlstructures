Provided Algorithms
===================

We provide multiple RL algorithms as examples.

1) A2C with General Advantage Estimator
2) PPO with discrete actions
3) Double Duelling Q-Learning + Prioritized Experience Replay
4) SAC for continuous actions

The algorithms can be used as examples to implement your own algorithms.
Typical execution is : `python main.py`

Note that all algorithms produced a tensorboard and a CSV output (see `config["logdir"]` in `main.py`)
