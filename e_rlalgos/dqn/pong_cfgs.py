(
    {"environment/env_name": "PongNoFrameskip-v4",
            "n_envs": 1,
            "max_episode_steps": 15000,
            "discount_factor": 0.99,
            "epsilon_greedy_max": 0.9,
            "epsilon_greedy_min": 0.01,
            "epsilon_min_epoch": [200000,1000000],
            "replay_buffer_size": [100000],
            "n_batches": 32,
            "use_duelling": [False,True],
            "use_double": [False,True],
            "lr": [3e-5,1e-5,3e-6],
            "update_target_epoch":[1000,2000],
            "n_evaluation_processes": 4,
            "verbose": True,
            "n_evaluation_envs": 4,
            "time_limit": 7200,
            "env_seed": 48,
            "clip_grad": [2.0],
            "learner_device": "cuda",
            "logdir":"/checkpoint/denoyer/pong"
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
            "initial_buffer_epochs": 600,
            "qvalue_epochs": 1,
            "batch_timesteps": 4,
            "n_processes": 4,
        }
        ,
        {
            "initial_buffer_epochs": 120,
            "qvalue_epochs": 1,
            "batch_timesteps": 20,
            "n_processes": 4,
        }
    ]
)
