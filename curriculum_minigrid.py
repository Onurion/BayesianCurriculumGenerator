import os
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoRemarkableImprovement, StopTrainingOnRewardThreshold
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.inference import VariableElimination
# from sklearn.manifold import TSNE
import openTSNE
# import torchvision.models as models
from sklearn.decomposition import PCA
from scipy.linalg import norm
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from bayes_utils import bayes_netzwerk
from utils import *
from encoder_utils import *
from gym_minigrid.envs.custom_doorkey import CustomDoorKeyEnv
from sklearn.preprocessing import KBinsDiscretizer
import warnings


def bayes_curriculum(args):
    N_eval_freq = args.n_eval_freq // args.n_procs
    N_sample = 500
    N_sample_init = 10000
    N_no_improvement = 400
    reward_threshold = 0.5
    N_levels = 3
    N_level_threshold = 3
    distance_coef = 2.0
    img_size = (224, 224)
    desired_column = "NormMapSim_Euclidean"
    map_thresh_coeff = 0.4
    # checked_maps = []
    # checked_maps_weight = 3
    env_type = args.env_type

    with open(args.map_file, 'rb') as handle:
        new_target_maps = pickle.load(handle)

    N_maps = len(new_target_maps)
    for j_map in range(N_maps):
        target_map = new_target_maps[j_map]
        
        for i_iter in range(args.N_train_iter):
            print ("Current Iter #" + str(i_iter) +" Env. Map #" + str(j_map) + "\n")
            print (target_map)

            autoencoder = DeepAutoencoder2()
            autoencoder.load_state_dict(torch.load(args.autoencoder_model))
            # Extract the encoder part of the trained model
            encoder = autoencoder.encoder

            bayes_model, empty_cells = bayes_netzwerk(target_map, env_type)

            inference = BayesianModelSampling(bayes_model)
            gen_data = inference.forward_sample(size=N_sample_init)
            gen_data['Reward'] = -100*np.ones(N_sample_init)
            encoder_output = []

            returned_data, total_maps, total_maps_items, map_images = eliminate_infeasible_maps(gen_data, target_map, empty_cells, env_type)
            map_indices = list(returned_data.index)
            print ("Map length for tSNE training: ", len(map_indices))
            img_orig_size = map_images[0].shape[0]

            N_train = len(list(total_maps.values()))
            X_train = np.array(map_images).reshape(N_train, img_orig_size, img_orig_size, 3)
            
            for i in range(20):
                encoder_output.append(get_latent_output(X_train[i], encoder, img_size, preprocess = False).ravel())

            target_map_image = get_env_img(target_map, env_type)

            encoder_output.append(get_latent_output(target_map_image, encoder, img_size, preprocess = False).ravel())
            # encoder_output.append(get_updated_latent_output(target_map, encoder))
            encoder_output_array = np.array(encoder_output).reshape(len(encoder_output), -1)


            # TSNE Model
            tsne_instance = openTSNE.TSNE(
                perplexity=100,
                initialization="pca",
                metric="euclidean",
                n_jobs=8,
                random_state=3
            )

            # encoded_out_embedded = tsne_model.fit(data_scaled)
            tsne_model = tsne_instance.fit(encoder_output_array)

            N_figure_list = []
            curriculum_items = []
            curriculum_maps = []
            curriculum_maps_images = []
            curriculum_costs = []
            curriculum_status = []
            generation_indices = []
            solved_maps = []
            solved_items = []
            # unsolved_maps = dict()

            # for element in element_lists:
            #     target_items.append(list(np.where(np.sum(empty_cells == np.concatenate(np.where(target_map == element)), axis=1) == 2)[0]))
            # target_items = list(np.array(target_items).ravel())

            # for i in range(15):
            #     unsolved_maps[i] = []   


            level = 0
            train_is_on = True
            while(train_is_on):
                current_folder = args.output_folder + "_iter_" + str(i_iter) + "_map_" + str(j_map) # + "/level_" + str(level)
                if not os.path.exists(current_folder):
                    os.makedirs(current_folder)

                discretizer = KBinsDiscretizer(n_bins=N_levels, encode='ordinal', strategy='uniform')
                inference = BayesianModelSampling(bayes_model)
                gen_data = inference.forward_sample(size=N_sample)
                gen_data['Reward'] = -100*np.ones(N_sample)
                encoder_output = []

                returned_data, total_maps, total_maps_items, map_images = eliminate_infeasible_maps(gen_data, target_map, empty_cells, env_type)
                map_indices = list(returned_data.index)

                N_train = len(list(total_maps.values()))
                X_train = np.array(map_images).reshape(N_train, img_orig_size, img_orig_size, 3)
                for i in range(N_train):
                    encoder_output.append(get_latent_output(X_train[i], encoder, preprocess = False).ravel())
                
                print (f"Level: {level}, Total map length: {len(map_indices)}")

                target_map_image = get_env_img(target_map, env_type)
                target_map_output = tsne_model.transform(get_latent_output(target_map_image, encoder).reshape(1,-1))
                target_output_array = np.repeat(target_map_output, len(encoder_output), axis=0)

                encoder_output_array = tsne_model.transform(np.array(encoder_output).reshape(N_train,-1))

                euclidean_similarities = euclidean_similarity(target_output_array, encoder_output_array)

                for k, index in enumerate(map_indices):           
                    returned_data.loc[index,"Encoder_Val"] = euclidean_similarities[k] # update the reward value with the obtained one

                sorted_data = returned_data.sort_values("Encoder_Val", ascending=True)
                sorted_data['NormEncoder_Val'] = sorted_data["Encoder_Val"] / np.max(sorted_data["Encoder_Val"].values)

                sorted_data['Euclidean_Val'] = sorted_data.apply(lambda row: calculate_element_distances_row(row, target_map, empty_cells, env_type, distance_coef), axis=1)
                sorted_data['NormEuclidean_Val'] = sorted_data["Euclidean_Val"] / np.max(sorted_data["Euclidean_Val"].values)

                sorted_data["EuclideanEncoder_Val"] = sorted_data['NormEuclidean_Val'] + (1 - sorted_data['NormEncoder_Val'])
                sorted_data["NormSimilarity"] = sorted_data["EuclideanEncoder_Val"] / np.max(sorted_data["EuclideanEncoder_Val"].values)

                sorted_data['Difficulty_EuclideanEncoder'] = discretizer.fit_transform(sorted_data['NormSimilarity'].values.reshape(-1, 1))

                possible_indices = list(sorted_data[sorted_data["Difficulty_EuclideanEncoder"] == level].index)
                probs = (1 - sorted_data[sorted_data["Difficulty_EuclideanEncoder"] == level]["NormSimilarity"].values)
                probs_normalized = probs / np.sum(probs)
                selected_map_index = np.random.choice(possible_indices, p=probs_normalized)
                # selected_map_index = random.sample(indices, 1)[0]
                map_items = sorted_data.loc[selected_map_index].values[0:3].astype(int)
                current_map = generate_map(map_items, target_map, empty_cells, env_type)
                current_map_image = get_env_img(current_map, env_type)

                current_map_output = tsne_model.transform(get_latent_output(current_map_image, encoder).reshape(1,-1))
                current_map_array = np.repeat(current_map_output, len(encoder_output), axis=0)
                euclidean_similarities = euclidean_similarity(current_map_array, encoder_output_array)
                
                
                curriculum_items.append(map_items)
                curriculum_maps.append(current_map)
                current_map_image = get_env_img(current_map, env_type)
                current_similarity_val = calculate_euclidian_distances(current_map, target_map, distance_coef)
                
                for k, index in enumerate(map_indices):           
                    sorted_data.loc[index,"MapSim_Encoder"] = euclidean_similarities[k] # update the reward value with the obtained one
                sorted_data["NormMapSim_Encoder"] = sorted_data["MapSim_Encoder"] / np.max(sorted_data["MapSim_Encoder"].values)

                sorted_data["MapSim_Euclidean"] = sorted_data.apply(lambda row: calculate_euclidian_row(row, current_map, empty_cells, env_type, distance_coef), axis=1)
                sorted_data["NormMapSim_Euclidean"] = sorted_data["MapSim_Euclidean"] / np.max(sorted_data["MapSim_Euclidean"].values)

                sorted_data["MapSimilarity"] = 0.4*sorted_data["NormMapSim_Encoder"] + 0.6*(1 - sorted_data["NormMapSim_Euclidean"])
                # sorted_data['NormPrev_Sim'] = sorted_data["Prev_Sim"] / np.max(sorted_data["Prev_Sim"].values)
                thresh_val = np.mean(sorted_data[desired_column].values) * map_thresh_coeff


                print (f"Target: {j_map}/{len(new_target_maps) - 1} - Map cost: {current_similarity_val:.5} - Level: {level}/{N_levels - 1}")
                print ("Current Map: \n", current_map)

                env_kwargs = {"visualization": False, "env_map": current_map}    # inner_map_lim = 6
                # train_env = make_atari_env(env_id=args.env_id, n_envs=args.n_procs, env_kwargs=env_kwargs, monitor_dir=current_folder + "/level_" + str(level) , vec_env_cls=DummyVecEnv)
                train_env = make_vec_env(env_id=args.env_id, n_envs=args.n_procs, env_kwargs=env_kwargs, monitor_dir=current_folder + "/level_" + str(level), vec_env_cls=DummyVecEnv)  
                train_env.reset()

                if level > 0:
                    model = args.algorithm.load(path=current_folder + "/saved_models/level_" + str(level - 1) + "_best/best_model", verbose=0, only_weights = False) # + "/best_model"
                    model.tensorboard_log = "./" + current_folder + "/level_" + str(level) + "/tensorboard/"
                    model.set_env(train_env)
                    print ("Best model from level #", (level - 1), " is uploaded!")
                    print ("The map, where the best previous model was trained, is: \n", solved_maps[-1] )
                elif args.load_folder != "":
                    model = args.algorithm.load(path=current_folder + "/saved_models/" + args.load_folder, verbose=0, only_weights = False) # + "/best_model"
                    model.tensorboard_log = "./" + current_folder + "/level_" + str(level) + "/tensorboard/"
                    model.set_env(train_env)
                    print ("Best model from load folder argument ", args.load_folder, " is uploaded!")
                else:
                    model = args.algorithm(args.policy, train_env, tensorboard_log= current_folder + "/level_" + str(level) + "/tensorboard/")
                
                stop_callback = StopTrainingOnNoRemarkableImprovement(max_no_improvement_evals = (level + 1) * N_no_improvement, reward_threshold=reward_threshold, verbose = 1)
                
                callback = EvalCallback(eval_env=train_env, callback_on_new_best=stop_callback, n_eval_episodes = args.n_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder + "/level_" + str(level) +"_log",
                                        best_model_save_path = current_folder + "/saved_models/level_" + str(level), deterministic=args.deterministic, verbose=1)
                
                model.learn(total_timesteps=args.train_timesteps, tb_log_name = current_folder  + "/level_" + str(level), callback=callback)
            
                # Load the model
                model = model.load(path=current_folder + "/saved_models/level_" + str(level) + "/best_model", verbose=0, only_weights = False) # + "/best_model"
                episode_rewards, episode_lengths = evaluate_policy(model, train_env, n_eval_episodes=args.n_eval_episodes, render=False, deterministic=args.deterministic, return_episode_rewards=True, warn=True)
                mean_reward_current = np.mean(episode_rewards)
                std_reward_current = np.std(episode_rewards)
                print (f"Mean reward in train env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
                train_env.close()

                checked_maps = sorted_data.loc[sorted_data[desired_column] < thresh_val].values

                solvable_task = False
                if mean_reward_current >= reward_threshold:
                    solvable_task = True
                    sorted_data.loc[sorted_data[desired_column] < thresh_val, "Difficulty_EuclideanEncoder"] = level
                    
                    solved_maps.append(current_map)
                    solved_items.append(map_items)
                    print (f"A solvable map is obtained for generation # {level} \n")
                    model.save(path=current_folder + "/saved_models/level_" + str(level) + "_best/best_model")
                else:
                    print ("This one can not be solved! \n")
                    sorted_data.loc[sorted_data[desired_column] < thresh_val, "Difficulty_EuclideanEncoder"] = level + 1
                    # unsolved_maps[level].append(current_map)
                
                curriculum_maps_images.append(current_map_image)
                curriculum_costs.append(sorted_data.loc[selected_map_index]["NormSimilarity"])
                curriculum_status.append(solvable_task)
                generation_indices.append(level)
                

                # Create a new DataFrame by duplicating the checked maps based on their weight
                # checked_maps_df = pd.concat([pd.DataFrame([row] * (checked_maps_weight - 1)) for row in checked_maps])
                # checked_maps_df.columns=sorted_data.columns

                # Append the checked maps DataFrame to the original sorted_maps DataFrame
                # weighted_maps = pd.concat([sorted_data, checked_maps_df], ignore_index=True)

                weighted_maps = sorted_data.copy()        
                
                if env_type == "KeyCorridor" or env_type == "DoorKey":
                    bayes_model = BayesianModel([('S', 'K'), ('K', 'G'), ('G', 'Difficulty_EuclideanEncoder')])
                elif env_type == "LavaGap":
                    bayes_model = BayesianModel([('S', 'G'), ('G', 'Difficulty_EuclideanEncoder')])

                prior_type = "BDeu" # or "K2", "Dirichlet"
                bayes_model.fit(weighted_maps, estimator=BayesianEstimator, prior_type=prior_type)        

                target_finetuning = False
                if solvable_task:
                    level = level + 1 #min(level + 1, N_levels - 1)
                    
                    similarity_to_target = calculate_euclidian_distances(solved_maps[-1], target_map, distance_coef=2.0)
                    mean_encoder_val = np.mean(sorted_data["Euclidean_Val"].values)
                    print ("similarity target: ", similarity_to_target, "mean encoder val: ", mean_encoder_val)
                    
                    if level >= N_level_threshold:
                        target_finetuning = True                

                if target_finetuning:
                    env_kwargs = {"visualization": False, "env_map": target_map}    # inner_map_lim = 6
                    target_env = make_atari_env(env_id=args.env_id, n_envs=args.n_procs, env_kwargs=env_kwargs, monitor_dir=current_folder, vec_env_cls=DummyVecEnv)

                    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
                    target_callback = EvalCallback(eval_env=target_env, callback_after_eval=stop_callback, n_eval_episodes = args.n_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                            best_model_save_path = current_folder + "/saved_models/level_" + str(level) + "_target", deterministic=args.deterministic, verbose=1)
                    
                    print ("\nFine tuning in the target environment")
                    print ("Target map: \n", target_map)
                    target_env.reset()
                    model.tensorboard_log = "./" + current_folder + "/level_" + str(level) + "/tensorboard_target/"
                    model.set_env(target_env)
                    model.learn(total_timesteps=args.train_timesteps, tb_log_name = current_folder + "/level_" + str(level) + "_target", callback=target_callback)
                    
                    model = model.load(path=current_folder + "/saved_models/level_" + str(level) + "_target" + "/best_model", verbose=0, only_weights = False) # + "/best_model"
                    episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=50, render=False, deterministic=args.deterministic, return_episode_rewards=True, warn=True)
                    mean_reward_target = np.mean(episode_rewards)
                    std_reward_target = np.std(episode_rewards)
                    print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} ")
                    target_env.close()


                    if mean_reward_target >= reward_threshold:
                        print (f"Task # {j_map} is complete!")
                        model = model.load(path=current_folder + "/saved_models/level_" + str(level) + "_target" + "/best_model", verbose=1, only_weights = False) # + "/best_model"
                        model.save(path=current_folder + "/saved_models/best/best_model")
                        curriculum_maps.append(target_map)
                        N_figures = output_map_file(total_maps=curriculum_maps, map_costs=curriculum_costs, status=curriculum_status, gen_num=generation_indices, gen=None, folder=current_folder, N_figure_list = N_figure_list, title= "Target_" + str(j_map) + "_Curriculum", curriculum_list=True, env_type=env_type)
                        N_figure_list.append(N_figures)
                        train_is_on = False
                        break

                with open(current_folder + "/variables.pickle", 'wb') as handle:
                    pickle.dump([curriculum_maps_images, curriculum_maps, curriculum_status], handle, protocol=pickle.HIGHEST_PROTOCOL)

                if not train_is_on:
                    break
                elif level >= N_levels and train_is_on:
                    N_levels += 1
                    print (f"Number of level is increasing! New number of levels: {N_levels}")


def vanilla_train(args):
    N_eval_freq = args.n_eval_freq // args.n_procs
    reward_threshold = 0.5

    print (f"Vanilla Training! \n")

    with open(args.map_file, 'rb') as handle:
        new_target_maps = pickle.load(handle)

    for j_map, env_map in enumerate(new_target_maps):
        for i_iter in range(args.N_train_iter):
    
            print ("Current Iter #" + str(i_iter) +" Env. Map #" + str(j_map) + "\n")
            print (env_map)

            current_folder = args.output_folder + "_iter_" + str(i_iter) + "_map_" + str(j_map)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            env_kwargs = {"visualization": args.visualization, "env_map": env_map}    # , "custom": False, "env_map": None

            # train_env = RGBImgObsWrapper_Obstructed(gym.make(args.env_id, **env_kwargs))
            train_env = make_atari_env(env_id=args.env_id, n_envs=args.n_procs, env_kwargs=env_kwargs, monitor_dir=current_folder, vec_env_cls=DummyVecEnv)
            train_env.reset()
            
            model = args.algorithm(args.policy, train_env, tensorboard_log= "./" + current_folder + "/tensorboard/")   

            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

            callback = EvalCallback(eval_env=train_env, callback_after_eval=stop_callback, n_eval_episodes = args.n_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder + "/log",
                                    best_model_save_path = current_folder + "/saved_models", deterministic=args.deterministic, verbose=1)

            
            model.learn(total_timesteps=args.train_timesteps, tb_log_name = "level_", callback=callback)     

            # Load the model
            model = model.load(path=current_folder + "/saved_models/best_model", verbose=0, only_weights = False) # + "/best_model"
            episode_rewards, episode_lengths = evaluate_policy(model, train_env, n_eval_episodes=args.n_eval_episodes, render=False, deterministic=args.deterministic, return_episode_rewards=True, warn=True)
            mean_reward_current = np.mean(episode_rewards)
            std_reward_current = np.std(episode_rewards)
            print (f"Mean reward in train env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
            train_env.close()