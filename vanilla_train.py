import os
import time
import pickle
import torch.nn as nn
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from env_util import make_vec_env
# from minigrid.wrappers import RGBImgObsWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# from utils import CustomCNN
# from encoder_utils import *

def vanilla_train(args):
    N_eval_freq = args.n_eval_freq // args.n_procs
    reward_threshold = 0.5

    print (f"Vanilla Training! \n")

    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )


    with open(args.map_file, 'rb') as handle:
        new_target_maps = pickle.load(handle)


    for j_map, env_map in enumerate(new_target_maps):
        for i_iter in range(args.N_train_iter):
    
            print ("Current Iter #" + str(i_iter) +" Env. Map #" + str(j_map) + "\n")
            print (env_map)

            current_folder = args.output_folder + "_iter_" + str(i_iter) + "_map_" + str(j_map)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            # env_kwargs = {"visualization": args.visualization, "env_map": env_map}    # , "custom": False, "env_map": None
            env_kwargs = {"env_map": env_map}

            # train_env = RGBImgObsWrapper_Obstructed(gym.make(args.env_id, **env_kwargs))
            # train_env = make_atari_env(env_id=args.env_id, n_envs=args.n_procs, env_kwargs=env_kwargs, monitor_dir=current_folder, vec_env_cls=DummyVecEnv)
            train_env = make_vec_env(env_id=args.env_id, n_envs=args.n_procs, env_kwargs=env_kwargs, monitor_dir=current_folder, vec_env_cls=DummyVecEnv)  
            train_env.reset()
            
            model = args.algorithm(args.policy, train_env, tensorboard_log= current_folder + "/tensorboard/")
            

            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

            callback = EvalCallback(eval_env=train_env, callback_on_new_best=stop_callback, n_eval_episodes = args.n_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder + "/log",
                                    best_model_save_path = current_folder + "/saved_models", deterministic=args.deterministic, verbose=1)

            
            model.learn(total_timesteps=args.train_timesteps, tb_log_name = "level_", callback=callback)     

            # Load the model
            model = model.load(path=current_folder + "/saved_models/best_model", verbose=0, only_weights = False) # + "/best_model"
            episode_rewards, episode_lengths = evaluate_policy(model, train_env, n_eval_episodes=args.n_eval_episodes, render=False, deterministic=args.deterministic, return_episode_rewards=True, warn=True)
            mean_reward_current = np.mean(episode_rewards)
            std_reward_current = np.std(episode_rewards)
            print (f"Mean reward in train env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
            train_env.close()