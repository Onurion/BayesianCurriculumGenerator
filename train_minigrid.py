import argparse
import pickle
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
from curriculum_minigrid import bayes_curriculum
from vanilla_train import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--env_id', default="MiniGrid-CustomDoorKey-v0", type=str, help='scenario') #MiniGrid-CustomDoorKey-8x8-v0, MiniGrid-CustomKeyCorridor-v0, MiniGrid-ObstructedMaze-2Dlhb-v0, MiniGrid-MultiRoom-N6-v0, MiniGrid-KeyCorridorS4R3-v0, MiniGrid-KeyCorridorS5R3-v0
    parser.add_argument('--output_folder', default="5Feb_DoorKey_Vanilla", type=str, help='output folder name')
    parser.add_argument('--env_type', default="DoorKey", type=str, help='name of the environment')
    parser.add_argument('--train_type', default="", type=str, help='name of the environment')
    parser.add_argument('--map_file', default="doorkey_all_maps.pkl", type=str, help='name of the policy')
    parser.add_argument('--autoencoder_model', default="autoencoder_doorkey/best_model.pth", type=str, help='encoder model to be utilized')
    parser.add_argument('--N_train_iter', default=2, type=int, help='train iteration')
    # parser.add_argument('--tsne_train_file', default="extracted_features_keycorridor_1_resnet152_normal.pickle", type=str, help='folder name')
    parser.add_argument('--train_timesteps', default=2000000, type=int, help='train timesteps')
    parser.add_argument('--seed', default=1, type=int, help='seed number for test')
    parser.add_argument('--n_eval_freq', default=400, type=int, help='evaluation interval')
    parser.add_argument('--n_eval_episodes', default=10, type=int, help='evaluation episodes')
    parser.add_argument('--n_procs', default=1, type=int, help='number of processes to execute')
    parser.add_argument('--deterministic', default = True, type=bool, help='how should the policy act')
    parser.add_argument('--algorithm', default=PPO, help='name of the algorithm')
    parser.add_argument('--policy', default="CnnPolicy", type=str, help='name of the policy')
    # parser.add_argument('--episode_length', default = 2500, type=int, help='episode length')
    parser.add_argument('--load_folder', default="", type=str, help='output folder name')
    parser.add_argument('--visualization', default = False, type=bool, help='to visualize')
    args = parser.parse_args()

    if args.train_type == "Vanilla":
        vanilla_train(args=args)
    else:
        bayes_curriculum(args=args)
    