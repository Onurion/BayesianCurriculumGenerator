import argparse
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
from Aerorival import Aerorival
from curriculum_aerorival import *
from vanilla_aerorival import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_timesteps', default=2000000, type=int, help='number of test iterations')
    parser.add_argument('--eval_freq', default=500, type=int, help='evaluation interval')
    parser.add_argument('--n_procs', default=4, type=int, help='number of processes to execute')
    parser.add_argument('--train_type', default="vanilla", type=str, help='model to be loaded')
    parser.add_argument('--seed', default=700, type=int, help='seed number for test')
    parser.add_argument('--algorithm', default=SAC, help='name of the algorithm')
    parser.add_argument('--env_id', default=Aerorival, help='name of the algorithm')
    parser.add_argument('--load_folder', default="", type=str, help='model to be loaded')
    parser.add_argument('--difficulty', default="easy", type=str, help='model to be loaded')
    parser.add_argument('--output_folder', default="Vanilla_SAC_Feb3_Hard", type=str, help='the output folder')
    parser.add_argument('--target_training', default = False, action='store_true', help='to start fine tuning in target environment')
    parser.add_argument('--visualization', default = False, action='store_true', help='to visualize')
    args = parser.parse_args()


    if args.train_type == "vanilla":
        noncurriculum_train(train_timesteps = args.train_timesteps, env_id=args.env_id, algorithm=args.algorithm, difficulty=args.difficulty, n_procs=args.n_procs,  eval_freq=args.eval_freq, seed = args.seed,  output_folder = args.output_folder,  visualization = args.visualization)
    else:
        curriculum_train(train_timesteps = args.train_timesteps, env_id=args.env_id, algorithm=args.algorithm, difficulty=args.difficulty, n_procs=args.n_procs, eval_freq=args.eval_freq, seed=args.seed, output_folder=args.output_folder, load_folder=args.load_folder, target_training=args.target_training, visualization=args.visualization)
    