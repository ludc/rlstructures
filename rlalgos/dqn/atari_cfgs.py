(
    {"environment/env_name": ["PongNoFrameskip-v4","AsteroidsNoFrameskip-v4","BoxingNoFrameskip-v4","SeaquestNoFrameskip-v4","TennisNoFrameskip-v4","JamesbondNoFrameskip-v4"],
            "n_envs": 1,
            "max_episode_steps": 15000,
            "discount_factor": 0.99,
            "epsilon_greedy_max": 0.99,
            "epsilon_greedy_min": [0.1,0.01],
            "epsilon_min_epoch": [200000,2000000],
            "replay_buffer_size": [100000],
            "n_batches": 32,
            "use_duelling": [False],
            "use_double": [True],
            "lr": [1e-4,3e-5],
            "n_evaluation_processes": 4,
            "verbose": True,
            "optim":["RMSprop","Adam"],
            "n_evaluation_envs": 4,
            "time_limit": 28800,
            "env_seed": 48,
            "clip_grad": [2.0],
            "learner_device": "cuda",

            "update_target_hard":[True],
            "update_target_epoch":1000,
            "update_target_tau": 0.005,

            "logdir":"/checkpoint/denoyer/atari",
            "save_every":1000,
    }
,
    [
        {
            "initial_buffer_epochs": 600,
            "qvalue_epochs": 1,
            "batch_timesteps": 4,
            "n_processes": 4,
            "buffer/alpha":0.,
            "buffer/beta":0.,
            "as_fast_as_possible":[False],
        }
        ,
        {
            "initial_buffer_epochs": 600,
            "qvalue_epochs": 4,
            "batch_timesteps": 4,
            "n_processes": 4,
            "buffer/alpha":0.0,
            "buffer/beta":0.0,
            "as_fast_as_possible":[False],
        }
        ,
        {
            "initial_buffer_epochs": 600,
            "qvalue_epochs": 1,
            "batch_timesteps": 20,
            "as_fast_as_possible":[True],
            "n_processes": 4,
            "buffer/alpha":0.0,
            "buffer/beta":0.0,
        }

    ]
)
