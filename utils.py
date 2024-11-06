
import numpy as np
import pandas as pd
import pickle
import itertools
import operator
from typing import List
from dstar import Dstar, Map
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from gym_minigrid.envs.custom_doorkey import CustomDoorKeyEnv
from gym_minigrid.envs.custom_keycorridor import CustomKeyCorridor
from gym_minigrid.envs.custom_lavagap import CustomLavaGapEnv
from encoder_utils import DeepAutoencoder2, get_latent_output
from sklearn.metrics.pairwise import cosine_similarity
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch as th
import keras
from keras import layers
from encoder_utils import *

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN without residual connections.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Convolutional layers
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Activation function
        self.relu = nn.ReLU()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            sample_output = self.forward_conv(sample_input)
            n_flatten = sample_output.shape[1]

        # Linear layers
        self.linear1 = nn.Linear(n_flatten, features_dim)
        self.linear2 = nn.Linear(features_dim, features_dim)

    def forward_conv(self, observations: th.Tensor) -> th.Tensor:
        x = self.relu(self.conv1(observations))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.flatten(x)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.forward_conv(observations)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

map_dict = {"free":0, "wall": 1,  "key":2, "door":3, "goal": 4, "start":5}
# elements_dict = {0:5, 1:2, 2:4}
elements_dict = {"S":5, "K":2, "G":4}

def euclidean_similarity(a, b):
    return np.linalg.norm(a - b, axis=1)

def manhattan_similarity(x, y):
    return np.sum(np.abs(x - y), axis=1)

def check_similarity(map1, map2, encoder, tsne_model):
    latent_out1 = get_latent_output(map1, encoder, preprocess = False).ravel()
    latent_out2 = get_latent_output(map2, encoder, preprocess = False).ravel()

    tsne_out1 = tsne_model.transform(latent_out1.reshape(1,-1))
    tsne_out2 = tsne_model.transform(latent_out2.reshape(1,-1))

    similarity = np.linalg.norm(tsne_out1 - tsne_out2)

    return similarity


def check_similarity_extractor(map1, map2, model, tsne_model, type="manhattan"):    
    preprocessed_map1_image = preprocess_image(map1)
    map1_features_extractor = extract_features(preprocessed_map1_image, model)
    map1_tsne_output = tsne_model.transform(map1_features_extractor.reshape(1,-1))

    preprocessed_map2_image = preprocess_image(map2)
    map2_features_extractor = extract_features(preprocessed_map2_image, model)
    map2_tsne_output = tsne_model.transform(map2_features_extractor.reshape(1,-1))

    if type == "euclidean":
        similarity = manhattan_similarity(map1_tsne_output, map2_tsne_output)
    elif type == "manhattan":
        similarity = euclidean_similarity(map1_tsne_output, map2_tsne_output)

    return similarity


def check_similarity_array(map1, map2, tsne_model):    
    map1_tsne_output = tsne_model.transform(map1.reshape(1,-1))
    map2_tsne_output = tsne_model.transform(map2.reshape(1,-1))

    # similarity = manhattan_similarity(map1_tsne_output, map2_tsne_output)
    similarity = cosine_similarity(map1_tsne_output, map2_tsne_output)[0]

    return similarity
    

def compute_threshold(previous_maps, encoder, tsne_model, factor=0.0):
    if not previous_maps:
        return -np.inf

    similarities = [check_similarity(map1, map2, encoder, tsne_model) for map1, map2 in itertools.combinations(previous_maps, 2)]
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    return mean_similarity - factor * std_similarity


def is_significantly_similar(generated_map, previous_maps, model, tsne_model, threshold):
    # threshold = compute_threshold(previous_maps, encoder, tsne_model, factor)

    for previous_map in previous_maps:
        if check_similarity_extractor(generated_map, previous_map, model, tsne_model) < threshold: # check_similarity(generated_map, previous_map, encoder, tsne_model) < threshold:
            return True
    return False

def calculate_element_distances(current_map, distance_coef=2.0, env_type = "DoorKey"):
    map_dict_keys = list(map_dict.keys())
    map_dict_vals = list(map_dict.values())

    if env_type == "DoorKey" or env_type == "KeyCorridor":
        to_be_calculated = [2, 4, 5] # Key, Goal, start
    elif env_type == "LavaGap":
        to_be_calculated = [4, 5] # Goal, Start


    combination_list = itertools.combinations(to_be_calculated, 2)

    for element1, element2 in combination_list:
        element1_current_map = np.concatenate(np.where(current_map == element1))
        element2_current_map = np.concatenate(np.where(current_map == element2))
        distance_current = np.linalg.norm(element1_current_map - element2_current_map)
        if not check_feasibility_two_maps(current_map, current_map, start=map_dict_keys[map_dict_vals.index(element1)], goal=map_dict_keys[map_dict_vals.index(element2)]):
            distance_current += distance_coef


    return distance_current

def calculate_element_distances_row(row, target_map, empty_cells, env_type, distance_coef=2.0):
    map_dict_keys = list(map_dict.keys())
    map_dict_vals = list(map_dict.values())

    to_be_calculated = [2, 4, 5]
    combination_list = itertools.combinations(to_be_calculated, 2)

    if env_type == "DoorKey" or env_type == "KeyCorridor":
        map_items = row.values[0:3].astype(int)
    current_map = generate_map(map_items, target_map, empty_cells, env_type)

    for element1, element2 in combination_list:
        element1_current_map = np.concatenate(np.where(current_map == element1))
        element2_current_map = np.concatenate(np.where(current_map == element2))
        distance_current = np.linalg.norm(element1_current_map - element2_current_map)
        if not check_feasibility_two_maps(current_map, current_map, start=map_dict_keys[map_dict_vals.index(element1)], goal=map_dict_keys[map_dict_vals.index(element2)]):
            distance_current += distance_coef

    return distance_current

def calculate_euclidian_distances(current_map, target_map, distance_coef=2.0):
    to_be_calculated = [2, 4, 5]
    distance_list = []
    map_dict_keys = list(map_dict.keys())
    map_dict_vals = list(map_dict.values())

    for element in to_be_calculated:
        element_current_map = np.concatenate(np.where(current_map == element))
        element_target_map = np.concatenate(np.where(target_map == element))
        distance = np.linalg.norm(element_target_map - element_current_map)
        
        if not check_feasibility_two_maps(current_map, target_map, start=map_dict_keys[map_dict_vals.index(element)], goal=map_dict_keys[map_dict_vals.index(element)]):
            distance += distance_coef

        distance_list.append(distance)

    norm_distance = np.linalg.norm(distance_list) 
    return norm_distance

def calculate_map_items_difference(row):
    distance_list = []
    map_items = row.values[0:3].astype(int)
    combination_list = itertools.combinations(map_items, 2)

    for element1, element2 in combination_list:
        distance_current = np.sum(np.abs(element1 - element2))
        distance_list.append(distance_current)

    max_distance = np.max(distance_list)
    return max_distance

def calculate_map_items_difference_target_row(row, target_items):
    distance_list = []
    map_items = row.values[0:3].astype(int)
    
    for i, element1 in enumerate(map_items):
        distance_current = np.sum(np.abs(element1 - target_items[i]))
        distance_list.append(distance_current)

    max_distance = np.max(distance_list)
    return max_distance

def calculate_map_items_difference_target(current_items, target_items):
    distance_list = []

    for i, element1 in enumerate(current_items):
        distance_current = np.sum(np.abs(element1 - target_items[i]))
        distance_list.append(distance_current)

    max_distance = np.max(distance_list)
    return max_distance

def calculate_euclidian_row(row, target_map, empty_cells, env_type, distance_coef=2.0):
    to_be_calculated = [2, 4, 5]
    distance_list = []
    map_dict_keys = list(map_dict.keys())
    map_dict_vals = list(map_dict.values())

    if env_type == "DoorKey" or env_type == "KeyCorridor":
        map_items = row.values[0:3].astype(int)

    current_map = generate_map(map_items, target_map, empty_cells, env_type)

    for element in to_be_calculated:
        element_current_map = np.concatenate(np.where(current_map == element))
        element_target_map = np.concatenate(np.where(target_map == element))
        distance = np.linalg.norm(element_target_map - element_current_map)
        
        if not check_feasibility_two_maps(current_map, target_map, start=map_dict_keys[map_dict_vals.index(element)], goal=map_dict_keys[map_dict_vals.index(element)]):
            distance += distance_coef

        distance_list.append(distance)

    norm_distance = np.linalg.norm(distance_list) 
    return norm_distance

def find_easy_map(sorted_data, target_map, empty_cells, env_type = "DoorKey"):
    cost_dict = dict()
    indices = list(sorted_data.index)
    for index in indices:
        if env_type == "DoorKey" or env_type == "KeyCorridor":
            current_map_items = sorted_data.loc[index].values[:3].astype(int).tolist()
        elif env_type == "LavaGap":
            current_map_items = sorted_data.loc[index].values[:2].astype(int).tolist()

        current_map = generate_map(current_map_items, target_map, empty_cells, env_type)
        cost = calculate_element_distances(current_map, env_type=env_type)
        cost_dict[index] = cost

    easy_map_index = list(dict(sorted(cost_dict.items(), key=lambda item: item[1])).keys())[0]
    return easy_map_index, cost_dict

def normalize_difficulty(row, sorted_data):
    level = row['Difficulty']
    level_indices = sorted_data[sorted_data['Difficulty'] == level].index
    if not level_indices.empty:
        min_difficulty = sorted_data.loc[level_indices, "Reward"].min()
        max_difficulty = sorted_data.loc[level_indices, "Reward"].max()
        return (row["Reward"] - min_difficulty) / (max_difficulty - min_difficulty)
    else:
        return np.nan
    
def calculate_pseudo_cost(row, target_map, empty_cells, env_type):
    if env_type == "DoorKey" or env_type == "KeyCorridor":
        map_items = row.values[:3].astype(int).tolist()

    env_map = generate_map(map_items, target_map, empty_cells, env_type)
    cost = calculate_element_distances(env_map, env_type=env_type)

    return cost

def check_feasibility(map, env_type="DoorKey"):
    current_map = np.copy(map)
    current_map[current_map == map_dict["door"]] = map_dict["wall"]
    rows, cols = np.where(current_map == map_dict["wall"])
    obstacles = np.concatenate([rows.reshape([-1, 1]), cols.reshape([-1, 1])], axis=1)
    start_p = np.concatenate(np.where(current_map == map_dict["start"]))

    if env_type == "DoorKey" or env_type == "KeyCorridor":
        key_p = np.concatenate(np.where(current_map == map_dict["key"]))
    elif env_type == "LavaGap":
        key_p = np.concatenate(np.where(current_map == map_dict["goal"]))

    map_sz = current_map.shape[0]
    m = Map(map_sz, map_sz)
    
    m.set_obstacle([(i, j) for i, j in zip(obstacles[:,1], obstacles[:,0])])

    start = m.map[start_p[0]][start_p[1]]
    end = m.map[key_p[0]][key_p[1]]
    dstar = Dstar(m)
    feasible, pos_list, action_list = dstar.run(start, end)
    # print ("Feasibility: ", feasible)
    return feasible


def check_feasibility_two_maps(map0, map1, start="start", goal="key"):
    current_map = np.copy(map0)
    target_map = np.copy(map1)
    current_map[current_map == map_dict["door"]] = map_dict["wall"]
    target_map[target_map == map_dict["door"]] = map_dict["wall"]
    rows, cols = np.where(current_map == map_dict["wall"])
    obstacles = np.concatenate([rows.reshape([-1, 1]), cols.reshape([-1, 1])], axis=1)
    start_p = np.concatenate(np.where(current_map == map_dict[start]))
    goal_p = np.concatenate(np.where(target_map == map_dict[goal]))

    map_sz = current_map.shape[0]
    m = Map(map_sz, map_sz)
    
    m.set_obstacle([(i, j) for i, j in zip(obstacles[:,1], obstacles[:,0])])

    start = m.map[start_p[0]][start_p[1]]
    end = m.map[goal_p[0]][goal_p[1]]
    dstar = Dstar(m)
    feasible, pos_list, action_list = dstar.run(start, end)
    # print ("Feasibility: ", feasible)
    return feasible

def generate_map_from_dataframe(generated_data, map_index, target_map, empty_cells, env_type):
    if env_type == "DoorKey" or env_type == "KeyCorridor":
        gen_map = generated_data.loc[map_index].values[:3].astype(int).tolist()
    elif env_type == "LavaGap":
        gen_map = generated_data.loc[map_index].values[:2].astype(int).tolist()#list(map(int, generated_data.loc[map_index].values[:-1]))

    columns = generated_data.drop(["Reward"], axis=1).columns.tolist()

    N_elements = len(gen_map)
    if len(np.unique(gen_map)) != N_elements:
        return []

    base_map = np.copy(target_map)
    base_map[base_map == map_dict["start"]] = 0
    base_map[base_map == map_dict["goal"]] = 0
    base_map[base_map == map_dict["key"]] = 0

    current_map = np.copy(base_map)
    for index, column in zip(gen_map, columns):
        current_map[empty_cells[index][0], empty_cells[index][1]] = elements_dict[column]

    return current_map

def generate_map(map_items, target_map, empty_cells, env_type="DoorKey"):
    N_elements = len(map_items)
    if len(np.unique(map_items)) != N_elements:
        return []
    
    if env_type == "KeyCorridor" or env_type == "DoorKey":
        columns=["S", "K", "G"]
    elif env_type == "LavaGap":
        columns=["S", "G"]

    base_map = np.copy(target_map)
    base_map[base_map == map_dict["start"]] = 0
    base_map[base_map == map_dict["goal"]] = 0
    base_map[base_map == map_dict["key"]] = 0

    current_map = np.copy(base_map)
    for index, column in zip(map_items, columns):
        current_map[empty_cells[index][0], empty_cells[index][1]] = elements_dict[column]

    return current_map

def eliminate_infeasible_maps(generated_data, target_map, empty_cells, env_type="DoorKey"):
    returned_data = generated_data.copy()
    map_indices = []
    element_list = []
    map_images = []
    total_maps =  dict()
    total_maps_items = dict()
    indices = list(generated_data.index)
    for map_index in indices:
        # print (generated_data.loc[map_index].values[:-1])
        # gen_element = generated_data.loc[map_index].values[:-1].astype(int).tolist()#list(map(int, generated_data.loc[map_index].values[:-1]))
        # gen_element = generated_data.drop(["Reward"], axis=1).loc[map_index].values.astype(int).tolist()
        if env_type == "DoorKey" or env_type == "KeyCorridor":
            gen_element = generated_data.loc[map_index].values[:3].astype(int).tolist()
        elif env_type == "LavaGap":
            gen_element = generated_data.loc[map_index].values[:2].astype(int).tolist()

        if gen_element in element_list:
            returned_data = returned_data.drop(map_index)
            continue
        
        element_list.append(gen_element)
        current_map = generate_map_from_dataframe(generated_data, map_index, target_map, empty_cells, env_type)
        if len(current_map) == 0:
            returned_data = returned_data.drop(map_index)
            continue

        feasible = check_feasibility(map=current_map, env_type=env_type)
        if not feasible:
            returned_data = returned_data.drop(map_index)
            continue

        map_indices.append(map_index)
        map_images.append(get_env_img(current_map, env_type))
        total_maps[map_index] = current_map
        total_maps_items[map_index] = gen_element

    return returned_data, total_maps, total_maps_items, map_images


def get_env_img(target, env_type="DoorKey"):
    size = target.shape[0]
    if env_type == "DoorKey":
        # env_config = {"seed":0, 
        #             "size":size + 2,
        #             "env_map":target, "custom":True,
        #             "visualization":False}
        env_kwargs = {"visualization": False, "env_map": target}
        # target_map = np.array([[0,0,5,0,1,0],[0,0,0,0,1,0],[0,0,0,0,3,0],[0,0,0,0,1,0],[0,0,2,0,1,0],[0,0,0,0,1,4]])
        env = CustomDoorKeyEnv(**env_kwargs)
    elif env_type == "KeyCorridor":
        env_kwargs = {"visualization": False, "env_map": target}
        env = CustomKeyCorridor(**env_kwargs)
    elif env_type == "LavaGap":
        env_kwargs = {"visualization": False, "env_map": target}
        env = CustomLavaGapEnv(**env_kwargs)


    img = env.render(mode = None, highlight=False)
    return img

# def save_multi_image(filename, fig_list):
#     with PdfPages(filename) as pdf:
#         for fig in fig_list:
#             fig.savefig(pdf, format='pdf')
#     return len(fig_list)

def save_multi_image(filename, N_figure_list):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    if len(N_figure_list) > 0:
        n_start = N_figure_list[-1]
    else:
        n_start = 0

    for fig in figs[n_start:]:
        fig.savefig(pp, format='pdf')
    pp.close()

    return len(figs)


def output_map_file(total_maps, map_costs, status, gen_num, gen, N_figure_list, title, curriculum_list=False, folder=None, env_type="DoorKey"):
    plt.rcParams["figure.figsize"] = [14.00, 8.00]
    plt.rcParams["figure.autolayout"] = True

    N_maps = len(total_maps)
    n_cols = 4
    max_rows = 3
    N_figures = 0
    

    N_loop = int(np.ceil(N_maps / (max_rows*n_cols)))

    ind = 0
    for _ in range(N_loop):
        if ind >= N_maps:
            break

        n_rows = int(np.minimum(np.ceil((N_maps - ind) / n_cols), max_rows))
        
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
        loop_ind = 0
        for row in ax:
            if n_rows > 1:
                for col in row:
                    if ind >= N_maps:
                        break
                    col.imshow(get_env_img(total_maps[ind], env_type))
                    if curriculum_list:
                        if ind == (N_maps - 1):
                            col.title.set_text("Target Map")
                        else:
                            col.title.set_text("Level: " + str(gen_num[ind]) + " Sim: " + str(round(map_costs[ind], 2)) + "\nSolved: " + str(status[ind]))
                    else:
                        col.title.set_text("Gen: " + str(gen) + " Map: " + str(ind) + " Sim: " + str(round(map_costs[ind],2)))
                        
                    ind += 1
                    loop_ind += 1
            else:
                if ind >= N_maps:
                    break
                row.imshow(get_env_img(total_maps[ind], env_type))
                if curriculum_list:
                    if ind == (N_maps - 1):
                        row.title.set_text("Target Map")
                    else:
                        row.title.set_text("Level: " + str(gen_num[ind]) + " Sim: " + str(round(map_costs[ind],2)) + "\nSolved: " + str(status[ind]))
                else:
                    row.title.set_text("Gen: " + str(gen) + " Map: " + str(ind) + " Sim: " + str(round(map_costs[ind],2)))
                ind += 1
                loop_ind += 1
 

        n_figs = n_cols * n_rows
        if loop_ind < (n_cols * n_rows):
            for i in range(n_cols - 1, 0, -1):
                if n_figs == loop_ind:
                    break
                if n_rows > 1:
                    fig.delaxes(ax[n_rows - 1][i])
                else:
                    fig.delaxes(ax[i])
                n_figs -= 1

    if folder == None:
        filename = title + ".pdf"
    else:
        filename =  folder + "/" + title + ".pdf"
        
    if curriculum_list:
        N_figures = save_multi_image(filename, N_figure_list)
    # elif gen == (N_generation - 1):
    #     save_multi_image(filename)

    return N_figures



def encoder_init(filepath:str, x_shape:int):
    encoding_dim = 80  
    train_dim = x_shape#3#100
    input_img = keras.Input(shape=(train_dim,))
    encoded = layers.Dense(640, activation='relu')(input_img)
    encoded = layers.Dense(480, activation='relu')(encoded)
    encoded = layers.Dense(320, activation='relu')(encoded)
    encoded = layers.Dense(160, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    decoded = layers.Dense(train_dim, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.load_weights(filepath)

    return encoder

def calculate_encoder_loss(encoder, opentsne_trainer, x_current, x_target):
    norm_coeff = 77
    x_current = np.reshape(np.array(x_current) / norm_coeff, (1, -1))
    x_target = np.reshape(np.array(x_target) / norm_coeff, (1, -1))
    current_enc = encoder.predict(x_current, verbose=0)
    target_enc = encoder.predict(x_target, verbose=0)    
    # norm_val = np.linalg.norm(target_enc - current_enc)

    current_out = opentsne_trainer.transform(current_enc.reshape(1,-1))
    target_out = opentsne_trainer.transform(target_enc.reshape(1,-1))
    norm_val =  np.linalg.norm(target_out - current_out)
    return norm_val


def create_dataset(sorted_data, curriculum_index, curriculum_reward=1.0, beta=2, sample_size=100):
    new_data = sorted_data.copy()
    new_data["Reward"] = new_data["Reward"] / (0.5 * np.max(new_data["Reward"].values))
    new_data.loc[curriculum_index, "Reward"] += curriculum_reward
    prob = np.exp(beta * new_data["Reward"])
    new_data["Prob"] = prob / np.sum(prob)
    output_list = np.random.choice(new_data.index, p=new_data["Prob"], size=sample_size)
    dataframe_new = new_data.loc[output_list, ["S", "K", "G"]].astype(int)
    return dataframe_new


# def create_and_sort_dataset(inference, target_map, gen, prev_cost, distance_coef, empty_cells, N_gen, N_sample):
#     gen_data = inference.forward_sample(size=N_sample)
#     gen_data['Reward'] = -100 * np.ones(N_sample)

#     returned_data, total_maps, total_maps_items, map_images = eliminate_infeasible_maps(gen_data, target_map, empty_cells)
#     map_indices = list(returned_data.index)
#     cost_dict, cosim_dict, diff_dict = {}, {}, {}

#     for index in map_indices:
#         current_map = total_maps[index]
#         map_cost, diff_cost, cosim_val = calculate_total_cost(current_map, target_map, distance_coef)
#         returned_data.loc[index, "Reward"] = -1 * map_cost
#         cost_dict[index], cosim_dict[index], diff_dict[index] = -1 * map_cost, cosim_val, diff_cost

#     sorted_data = returned_data.sort_values("Reward", ascending=False).copy()

#     reward_list = sorted_data["Reward"].values
#     reward_mean, reward_median, reward_std = np.mean(reward_list), np.median(reward_list), np.std(reward_list)
#     coeff1, coeff2 = reward_mean + 2 * reward_std, reward_mean + 3 * reward_std

#     possible_indices = sorted_data.loc[(sorted_data["Reward"] > coeff1) & (sorted_data["Reward"] < coeff2) & (sorted_data["Reward"] > prev_cost)].index.tolist()

#     while len(possible_indices) == 0:
#         coeff1, coeff2 = coeff1 - abs(coeff1) * 0.1, coeff2 + abs(coeff2) * 0.1
#         prev_cost = prev_cost - abs(prev_cost) * 0.1
#         possible_indices = sorted_data.loc[(sorted_data["Reward"] > coeff1) & (sorted_data["Reward"] < coeff2) & (sorted_data["Reward"] > prev_cost)].index.tolist()

#     return possible_indices, sorted_data, cost_dict, cosim_dict, diff_dict


# def create_and_sort_dataset(inference, target_map, curriculum_maps_items, gen, prev_cost, encoder, opentsne_trainer, x_target, empty_cells, N_gen, N_sample):
#     generated_maps = []
#     gen_data = inference.forward_sample(size=N_sample)
#     gen_data['Reward'] = -100*np.ones(N_sample)
#     cost_dict = dict()
    
#     std_coeffs = np.linspace(1, -1, N_gen + 1)

#     returned_data, total_maps, total_maps_items = eliminate_infeasible_maps(gen_data, target_map, empty_cells)
#     map_indices = list(returned_data.index)

#     for index in map_indices:
#         current_map = total_maps[index]   
#         x_current = total_maps_items[index]                
#         generated_maps.append(current_map)
#         map_cost = calculate_encoder_loss(encoder, opentsne_trainer, x_current, x_target)
#         # map_cost, diff_cost, cosim_val = calculate_total_cost(current_map, target_map, distance_coef)
#         returned_data.loc[index,"Reward"] = map_cost # update the reward value with the obtained one

        
#     sorted_data = returned_data.copy()          

#     # rand_val = sorted_data.loc[random_index]["Reward"]
#     # N_range = len(sorted_indices) / N_generation
#     # ind_start = len(sorted_indices) - int(N_range * (gen+1))
#     # ind_end = len(sorted_indices) - int(N_range * gen)

#     sorted_data = sorted_data.sort_values("Reward", ascending=False)#.iloc[0:N_maps_update]
#     sorted_data["Reward"] = sorted_data["Reward"] / (0.5*np.max(sorted_data["Reward"].values))
#     sorted_indices = list(sorted_data.index)

#     for index in map_indices:
#         cost_dict[index] = sorted_data.loc[index,"Reward"]

#     reward_list = list(sorted_data["Reward"])
#     reward_list = np.array(reward_list)
#     reward_mean = np.mean(reward_list)
#     reward_median = np.median(reward_list)
#     reward_std = np.std(reward_list)

#     current_threshold = reward_mean + std_coeffs[gen]*reward_std

#     if gen == 0:
#         indices = sorted_data["Reward"] > current_threshold
#     elif gen == (N_gen - 1):
#         indices = sorted_data["Reward"] < current_threshold
#     else:
#         prev_threshold = reward_mean + std_coeffs[gen - 1]*reward_std
#         indices = (sorted_data["Reward"] > current_threshold) & (sorted_data["Reward"] < prev_threshold)

#     # possible_indices = list(sorted_data.loc[indices].index)

#     if len(curriculum_maps_items) > 0:
#         new_dataframe = sorted_data.loc[indices].copy()
#         for index, element in zip(list(sorted_data.loc[indices].index), sorted_data.loc[indices].values):
#             x_current = np.array(element[:-1]).astype(int)
#             map_curriculum = np.array(curriculum_maps_items[gen - 1].values[:-1]).astype(int)              
#             map_cost = calculate_encoder_loss(encoder, opentsne_trainer, x_current, map_curriculum)
#             # print ("current: ", x_current, "cost list: ", map_cost)
#             new_dataframe.loc[index, "Reward"] = map_cost
#         new_dataframe = new_dataframe.sort_values("Reward", ascending=False)
#         mean_val = np.mean(new_dataframe["Reward"].values)
#         updated_indices = new_dataframe["Reward"] > (mean_val / 2.0)
#         possible_indices = list(new_dataframe.loc[updated_indices].index)
#     else:
#         possible_indices = list(sorted_data.loc[indices].index)

#     # coeff1 = reward_mean + std_coeffs[gen] * reward_std
#     # coeff2 = reward_mean + std_coeffs[gen + 1] * reward_std
    
#     # possible_indices = list(sorted_data.loc[(sorted_data["Reward"] > coeff1) & (sorted_data["Reward"] < coeff2) & (sorted_data["Reward"] > prev_cost)].index)

#     # while len(possible_indices) == 0:
#     #     coeff1 = coeff1 - abs(coeff1) * 0.1
#     #     coeff2 = coeff2 + abs(coeff2) * 0.1
#     #     prev_cost = prev_cost - abs(prev_cost) * 0.1
#     #     possible_indices = list(sorted_data.loc[(sorted_data["Reward"] > coeff1) & (sorted_data["Reward"] < coeff2) & (sorted_data["Reward"] > prev_cost)].index)
        
#     return possible_indices, sorted_data, cost_dict


# def create_dataset(sorted_data, curriculum_index, curriculum_reward=1.0, beta=2):
#    new_data = sorted_data.copy()
#    new_data["Reward"] = new_data["Reward"] / (0.5*np.max(new_data["Reward"].values))
#    new_data.loc[curriculum_index, "Reward"] += curriculum_reward # the map with the highest loss will be trained
#    indices = list(new_data.index)
#    # new_data.loc[indices[0],"Reward"] += 1
#    new_data["Prob"] = 1*np.ones(len(indices))
#    for index in indices:
#       new_data.loc[index,"Prob"] = np.exp(beta*new_data.loc[index,"Reward"])

#    new_data.loc[:,"Prob"] /= np.sum(new_data.loc[:,"Prob"].values)
#    output_list = np.random.choice(indices, p=new_data.loc[:,"Prob"], size=100)

#    dataframe_new = []
#    for index in output_list:
#       dataframe_new.append(list(new_data.loc[index].values[0:3]))
#    dataframe_new = pd.DataFrame(dataframe_new,columns=["S","K","G"]).astype('int')
   
#    return dataframe_new



