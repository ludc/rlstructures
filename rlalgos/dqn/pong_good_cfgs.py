(
    {"environment/env_name": "PongNoFrameskip-v4",
            #n environment per process
            "n_envs": [4],
            #Maximum number of timesteps (for evaluation episodes)
            "max_episode_steps": 15000,
            #Discount factor
            "discount_factor": 0.99,
            #Epsilon value at the beginning of the learning process
            "epsilon_greedy_max": 1.0,
            #Epsilon value at the end of the decay
            "epsilon_greedy_min": [0.01],
            #Learning epoch when the min epsilon value is reached
            "epsilon_min_epoch": [100000],
            # Size of the replay buffer (number of transitions)
            "replay_buffer_size": [100000],
            # Size of the learning batches
            "n_batches": [32],
            #Duelling architecture ?
            "use_duelling": [False],
            #Double DQN ?
            "use_double": [True],
            #Learning rate
            "lr": [1e-4],
            #N env per evaluation process:   n_evaluation_processes*n_evaluation_envs is the number of trajectories used for each evaluation
            "n_evaluation_envs": 1,
            #N paralellel evaluation processes: increase if you wnat more frequent evaluation
            "n_evaluation_processes": 4,

            "verbose": False,
            #Optimizer name
            "optim":["Adam"],
            #Stop the process after n seconds
            "time_limit": 28800,
            #Env seed
            "env_seed": 48,
            #Clip gradient
            "clip_grad": [0.0],
            #cpu or cuda
            "learner_device": "cuda",
            # if True, the system will do as many model updates as possible
            "as_fast_as_possible":[True],

            #Update target model is a hard way (by copying) or with smooth upddates. If True=>update_target_epoch if False=>update_target_tau
            "update_target_hard":[True],
            #Copy to target model every n epochs
            "update_target_epoch":100,
            #Smooth update coefficient of the target model (i.e 0.005)
            "update_target_tau": 0.005,

            "logdir":"/checkpoint/denoyer/pong",
            #Save logged values every n timesteps (to avoid too large log files)
            "save_every":100,
    }
,
    [
        {
            "initial_buffer_epochs": 300,
            "qvalue_epochs": 1,
            "batch_timesteps": 20,
            "n_processes": 4,
            "buffer/alpha":0.0,
            "buffer/beta":0.0,
        }

    ]
)