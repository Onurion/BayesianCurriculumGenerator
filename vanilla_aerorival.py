import os
import gym
from stable_baselines3.common.callbacks import EvalCallback, EvaluateMeanRewardCallback
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from typing import Union, Callable, Type

def noncurriculum_train(train_timesteps: int, env_id:Callable[[], gym.Env], algorithm:Type[Union[PPO, SAC, TD3, DDPG, A2C]], difficulty:str, n_procs: int, output_folder: str, eval_freq:int, seed:int, visualization: bool):

    model_dir = output_folder + "/saved_models"
    print ("\nNon-Curriculum Approach")
    
    zone_prob = 1.0
    target_distance_coeff = 1.0
    target_positions = [[40, -40], [40,40]]
    agent_init_position = [-30, -30]
    rocket_velocity = 5.0
    enemy_velocity = 5.0
    n_eval_episodes = 50
    deterministic = False
    reward_threshold = 0.7

    if difficulty == "hard":
        n_zones = 3
    elif difficulty == "easy":
        n_zones = 1

    env_kwargs = {"n_zones": n_zones, "zone_prob": zone_prob, "target_distance_coeff": target_distance_coeff, "target_positions": target_positions, 
                  "agent_init_position": agent_init_position, "enemy_velocity": enemy_velocity, "rocket_velocity": rocket_velocity, "seed":seed, "visualization": visualization, "difficulty": difficulty}

    current_folder = output_folder + "/" 
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    train_env = make_vec_env(env_id=env_id, n_envs=n_procs, monitor_dir=current_folder + "/train", env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env(env_id=env_id, n_envs=n_procs, monitor_dir=current_folder + "/eval", env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv )

    after_callback = EvaluateMeanRewardCallback(eval_env=eval_env, algorithm=algorithm, model_path=model_dir, n_eval_episodes=n_eval_episodes, reward_threshold=reward_threshold, deterministic=deterministic)

    train_env.reset()
    model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
        

    
    eval_callback = EvalCallback(eval_env=eval_env, callback_on_new_best=after_callback,
                                         n_eval_episodes = n_eval_episodes,
                                         best_model_save_path= model_dir + "/",
                                         log_path=output_folder , eval_freq=eval_freq,
                                         deterministic=deterministic, render=False)
    

    model.learn(total_timesteps=train_timesteps, tb_log_name= "", callback=eval_callback)
    # model.save(model_dir + "/last_model_" + level_folder)

    train_env.close()
    eval_env.close()
