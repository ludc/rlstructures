(
    {"env_name": "PongNoFrameskip-v4",
            "n_envs": 1,
            "max_episode_steps": 15000,
            "env_seed": 42,
            "n_processes": 8,
            "n_evaluation_processes": 4,
            "n_evaluation_envs": 1,
            "time_limit": 43200,
            "lr": [0.001,0.0001],
            "discount_factor": 0.99,
            "critic_coef": 1.0,
            "entropy_coef": [0.01,0.001],
            "a2c_coef": 0.1,
            "gae_coef":[0.0,0.3,1.0],
            "logdir":"/checkpoint_denoyer/pong_a2c",
            "clip_grad":40,
            "learner_device":"cuda",
            "save_every":100
    }
,
    [
        {
            "a2c_timesteps": 20,
        }
        ,
        {
            "a2c_timesteps": 10,
        }
        ,
        {
            "a2c_timesteps": 40,
        }
        ,
    ]
)
