import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import numpy as np
from stable_baselines3 import PPO,SAC,TD3, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, EvaluateMeanRewardCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Union, Callable, Type
import gym

tfd = tfp.distributions

lower_bounds = np.array([0.1, 0.1, 0.1])
n_samples = 3
num_levels = 6

target_samples = [[1.0, 5.0, 5.0]]

def evaluate_model(eval_env, model, n_eval_episodes ): #
    # eval_env = env_id(**env_kwargs)
    episode_rewards = []
    episode_lengths = []
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        iteration = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
            iteration += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(iteration)

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    return mean_reward
    
def generate_sample(current_mean, current_std, target_sample, current_level):
    def model():
        # yield tfd.Normal(loc=current_mean[0], scale=current_std[0], name="zone_prob")
        yield tfd.Normal(loc=current_mean[0], scale=current_std[0], name="target_distance_coeff")
        yield tfd.Normal(loc=current_mean[1], scale=current_std[1], name="enemy_velocity")
        yield tfd.Normal(loc=current_mean[2], scale=current_std[2], name="rocket_velocity")


    # Create the joint distribution
    joint_dist = tfd.JointDistributionCoroutineAutoBatched(model)

    # Generate 100 samples
    generated_samples = joint_dist.sample(10000)

    # Convert the StructTuple to a numpy array for easier processing
    sample_array = np.stack([generated_samples.target_distance_coeff, 
                             generated_samples.enemy_velocity, 
                             generated_samples.rocket_velocity], axis=1) #generated_samples.zone_prob,
    

    # Filter out samples that violate the range rule
    filtered_samples = np.array([sample for sample in sample_array 
                        if np.all((sample >= lower_bounds) & (sample <= target_sample))])


    print ("Length of generated samples: ", len(filtered_samples))

    # Calculate Euclidean distances for the filtered samples
    euclidean_distances = np.linalg.norm(filtered_samples - target_sample, axis=1)

    inverted_distances = 1 / (euclidean_distances + 1e-6)  # Adding a small constant to avoid division by zero

    # Normalize to create a probability distribution
    probabilities = inverted_distances / np.sum(inverted_distances)


    # Sort samples by Euclidean distance in ascending order
    sorted_indices = np.argsort(inverted_distances)
    sorted_samples = filtered_samples[sorted_indices]

    samples_per_level = np.clip(len(sorted_samples) // num_levels, 1, 10000)

    # Classify sorted samples into levels
    level_indices = {level: [] for level in range(num_levels)}

    for i, index in enumerate(sorted_indices):
        level = i // samples_per_level
        if level >= num_levels:
            level = num_levels - 1
        level_indices[level].append(index)

    # Extract probabilities for level samples
    level_probabilities = probabilities[level_indices[current_level]]
    level_probabilities /= np.sum(level_probabilities)  # Normalize probabilities for level 2

    # Randomly select a sample index from level
    selected_index = np.random.choice(level_indices[current_level], p=level_probabilities)
    selected_sample = filtered_samples[selected_index]
    eucl_dist = np.linalg.norm(selected_sample - target_sample)

    return selected_sample, eucl_dist

def update_model(mean, std, target, learning_rate=0.1):
    """ Update model parameters """
    new_mean = mean + learning_rate * (target - mean)
    new_std = std + learning_rate * (np.abs(target - mean) - std)  # Adjust std deviation
    new_std = np.clip(new_std, 0.1, 5)  # Limit the std deviation to reasonable values
    return new_mean, new_std


def curriculum_train(train_timesteps: int, env_id:Callable[[], gym.Env], algorithm:Type[Union[PPO, SAC, TD3, DDPG, A2C]], difficulty:str, n_procs:int, eval_freq:int, seed:int, output_folder:str, load_folder:str, target_training:bool, visualization:bool):
    target_positions = [[40, -40], [40,40]]
    agent_init_position = [-30, -30]
    reward_threshold = 0.7
    

    target_sample = target_samples[0]

    base_folder = "./" + output_folder
    taken_samples = dict()
    distance_list = dict()
    current_level = 0
    current_level_sample = 0
    n_eval_episodes = 50
    deterministic = False
    
    continue_training = True
    target_mission = False

    current_mean = np.array([0.0, 0.0, 0.0])
    current_std =  np.array([0.1, 0.1, 0.1])

    if difficulty == "easy":
        n_zones = 1
    elif difficulty == "hard":
        n_zones = 3

    print("\n Folder: ", output_folder, " \n")

    for _ in range(num_levels):
        taken_samples[_] = []
        distance_list[_] = []

    # Define the joint distribution using a generator function
    def model():
        # yield tfd.Normal(loc=0.0, scale=0.1, name="zone_prob")
        yield tfd.Normal(loc=0.0, scale=0.1, name="target_distance_coeff")
        yield tfd.Normal(loc=0.0, scale=0.1, name="enemy_velocity")
        yield tfd.Normal(loc=0.0, scale=0.1, name="rocket_velocity")


    if os.path.exists(load_folder) and target_training:
        target_mission = True
        

    while(continue_training):

        if not target_mission:
            selected_sample, eucl_dist = generate_sample(current_mean, current_std, target_sample, current_level)

            sample = np.copy(selected_sample)
            folder = base_folder + "/level_" + str(current_level) + "_sample_" + str(current_level_sample)
            stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=250)
            

            env_kwargs = {"n_zones": n_zones, "zone_prob": 1.0, "target_distance_coeff": sample[0], "target_positions": target_positions, 
                            "agent_init_position": agent_init_position, "enemy_velocity": sample[1], "rocket_velocity": sample[2], "seed":seed, "visualization": visualization}
            
            print(f"\nLevel {current_level} Sample Index: {current_level_sample} Euclidean: {eucl_dist:.5f}")
            print(f"target_distance_coeff: {sample[0]:.4f} enemy_velocity: {sample[1]:.4f} rocket_velocity:{sample[2]:.4f} \n")

            env = make_vec_env(env_id, n_envs=n_procs, env_kwargs=env_kwargs, monitor_dir=folder + "/train", vec_env_cls=DummyVecEnv)
            # env = env_id(**env_kwargs)
            
            # Setup the evaluation callback
            eval_env = make_vec_env(env_id, n_envs=n_procs, env_kwargs=env_kwargs, monitor_dir=folder + "/eval", vec_env_cls=DummyVecEnv)
            after_callback = EvaluateMeanRewardCallback(eval_env=eval_env, algorithm=algorithm, model_path=folder, n_eval_episodes=n_eval_episodes, reward_threshold=reward_threshold, deterministic=deterministic)

            if current_level_sample > 0:
                rl_model = algorithm.load(path=base_folder + "/level_" + str(current_level) + "_sample_" + str(current_level_sample - 1) + "/best_model", verbose=0) # + "/best_model" only_weights = False
                rl_model.tensorboard_log = folder + "/tensorboard/"
                rl_model.set_env(env)
                print (f"\nBest model from level #{current_level} and sample index #{current_level_sample - 1} is uploaded!")
            elif current_level > 0:
                best_sample_index = np.argsort(distance_list[current_level - 1])[0]
                print ("\nEuclidean distances from previous level samples: ", distance_list[current_level - 1])
                rl_model = algorithm.load(path=base_folder + "/level_" + str(current_level - 1) + "_sample_" + str(best_sample_index) + "/best_model", verbose=0) # + "/best_model" only_weights = False
                rl_model.tensorboard_log = folder + "/tensorboard/"
                rl_model.set_env(env)
                print (f"Best model from level #{current_level} and sample index #{best_sample_index} is uploaded!")
            else:
                file_path = folder + "/best_model.zip"
                # Check if the file exists
                if os.path.exists(file_path):
                    rl_model = algorithm.load(path= file_path, verbose=1) # + "/best_model" only_weights = False
                    rl_model.tensorboard_log = folder + "/tensorboard/"
                    rl_model.set_env(env)
                    print ("\nA model has been found in directory: ", file_path, " and loaded succesfully!")
                else:
                    rl_model = algorithm("MlpPolicy", env, tensorboard_log= folder + "/tensorboard/")
                    print ("\nA new model has been created for training! Good luck!")

                
            eval_callback = EvalCallback(eval_env, callback_on_new_best=after_callback,
                                         callback_after_eval=stop_callback, 
                                         n_eval_episodes = n_eval_episodes,
                                         best_model_save_path= folder + "/",
                                         log_path=folder , eval_freq=eval_freq,
                                         deterministic=deterministic, render=False)

            # Train the agent
            rl_model.learn(total_timesteps=train_timesteps, callback=eval_callback) # 10000000
            rl_model = algorithm.load(path= folder + "/best_model", verbose=0)
            episode_rewards, episode_lengths = evaluate_policy(model=rl_model, env=eval_env, n_eval_episodes=n_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True)
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)

            # mean_reward = np.random.randint(-200, -100)

            print (f"Eval  Level: {current_level} Mean Reward: {mean_reward:.5f}\n")

            if mean_reward >= reward_threshold / 2.0 and current_level < num_levels :
                taken_samples[current_level].append(selected_sample)
                distance_list[current_level].append(eucl_dist)
                current_level_sample += 1

            
            if len(taken_samples[current_level]) >= n_samples:
                sample_mean = np.mean(taken_samples[current_level], axis=0)
                sample_std = np.std(taken_samples[current_level], axis=0)

                # Adjust model parameters
                current_mean, current_std = update_model(sample_mean, sample_std, target_sample)

                # Print updated parameters for monitoring
                print(f"\nThe Bayesian Model has been updated!")
                print(f"Updated Mean: {current_mean}")
                print(f"Updated Std: {current_std}\n")

                if current_level < (num_levels - 1):
                    current_level += 1
                    current_level_sample = 0
                    print ("Level of the curriculum is increasing! New level: ", current_level)
            
            if current_level >= (num_levels - 1) and len(taken_samples[current_level]) >= n_samples and mean_reward >= reward_threshold / 2.0:
                target_mission = True
                print ("Conditions are fulfilled! Training for target mission has been initiated!")
                

        else:
            print ("\nFine-Tuning in Target Mission")
            folder = base_folder + "/target"
            sample = np.copy(target_sample)
            stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=1000)

            env_kwargs = {"n_zones": n_zones, "zone_prob": 1.0, "target_distance_coeff": target_sample[0], "target_positions": target_positions, 
                            "agent_init_position": agent_init_position, "enemy_velocity": target_sample[1], "rocket_velocity": target_sample[2], "seed":seed, "visualization": visualization}

            env = make_vec_env(env_id, n_envs=n_procs, env_kwargs=env_kwargs) #Acrobot 
            # env = env_id(**env_kwargs)

            # Setup the evaluation callback
            eval_env = make_vec_env(env_id, n_envs=n_procs, env_kwargs=env_kwargs)
            after_callback = EvaluateMeanRewardCallback(eval_env=eval_env, algorithm=algorithm, model_path=folder, n_eval_episodes=n_eval_episodes, reward_threshold=reward_threshold, deterministic=deterministic)

            if os.path.exists(load_folder) and target_training:
                print ("Best model from ", load_folder, " uploaded!")
                rl_model = algorithm.load(path=load_folder + "/best_model", verbose=0)
            else:
                rl_model = algorithm.load(path=base_folder + "/level_" + str(current_level) + "_sample_" + str(current_level_sample - 1) + "/best_model", verbose=0) # + "/best_model"
            
            rl_model.tensorboard_log = folder + "/tensorboard/"
            rl_model.set_env(env)
            print ("Best model from level #", (current_level), " is uploaded!")
            
                
            eval_callback = EvalCallback(eval_env, callback_on_new_best=after_callback, 
                                         callback_after_eval=stop_callback, 
                                         n_eval_episodes = n_eval_episodes,
                                         best_model_save_path= folder + "/",
                                         log_path=folder , eval_freq=eval_freq,
                                         deterministic=deterministic, render=False)

            # Train the agent
            rl_model.learn(total_timesteps=train_timesteps, callback=eval_callback) # 10000000
            rl_model = algorithm.load(path= folder + "/best_model", verbose=1)
            episode_rewards, episode_lengths = evaluate_policy(model=rl_model, env=eval_env, n_eval_episodes=n_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True)
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)

            # mean_reward = evaluate_model(env, rl_model, n_eval_episodes)

            # mean_reward = -200

            print (f"Eval Target Mean Reward: {mean_reward:.5f}")
           
            if mean_reward >= reward_threshold / 2.0: #fine tuning is successful, terminate the training
                continue_training = False
                rl_model.save(path=folder + "/best/best_model")
            elif mean_reward < reward_threshold / 2.0 and target_mission: #fine tuning is not successful, increase the level numbers
                target_mission = False
                print(f"Fine tuning for target mission is unsuccesful. The process goes on with level {current_level}\n" )


        env.close()
        eval_env.close()