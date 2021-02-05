(
    {"environment/env_name": "PongNoFrameskip-v4",
            "n_envs": 1,
            "max_episode_steps": 10000,
            "discount_factor": 0.99,
            "epsilon_greedy_max": 0.9,
            "epsilon_greedy_min": 0.01,
            "epsilon_min_epoch": [1000000,200000],
            "replay_buffer_size": [100000],
            "n_batches": 32,
            "use_duelling": [True],
            "use_double": [False,True],
            "lr": [0.0001,0.00003],
            "update_target_epoch":[200,1000],
            "n_evaluation_processes": 4,
            "verbose": True,
            "n_evaluation_envs": 4,
            "time_limit": 28800,
            "env_seed": 48,
            "clip_grad": [10.0],
            "learner_device": "cuda",
            "logdir":"./results"
    }
,
    [
        {
            "initial_buffer_epochs": 10000,
            "qvalue_epochs": 1,
            "batch_timesteps": 1,
            "n_processes": 1,
        }
        ,
        {
            "initial_buffer_epochs": 2500,
            "qvalue_epochs": 1,
            "batch_timesteps": 1,
            "n_processes": 4,
        }
        ,
        {
            "initial_buffer_epochs": 250,
            "qvalue_epochs": 10,
            "batch_timesteps": 10,
            "n_processes": 4,
        }
        ,

    ]
)
