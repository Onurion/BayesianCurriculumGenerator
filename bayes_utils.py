import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling


# 0-Empty, 1-Wall, 2-Key, 3-Door, 4-Goal, 5-Start

single_depend = [[0.5, 0.5, 0.6, 0.6, 0.6, 0.6],
                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.0, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.0, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.0, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]]

double_depend = [[0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.6, 0.6, 0.7, 0.6, 0.7, 0.7, 0.6, 0.6, 0.7, 0.7, 0.6, 0.7, 0.6, 0.6, 0.7, 0.7, 0.7, 0.6],
                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def bayes_netzwerk(target_map, env_type="KeyCorridor"):
    map_size = target_map.shape[0]
    current_map = np.copy(target_map)
    current_map[current_map == 5] = 0
    current_map[current_map == 4] = 0
    current_map[current_map == 2] = 0
    rows,cols = np.where(current_map == 0)
    empty_cells = np.concatenate([rows.reshape(-1,1), cols.reshape(-1,1)], axis=1)
    N_Cells = len(empty_cells)

    K_dependency = np.ones((N_Cells, N_Cells)) * 1.0/(N_Cells-1)
    for i in range(N_Cells):
        K_dependency[i][i] = 0

    G_dependency = np.zeros((N_Cells, N_Cells**2))
    depend_col = []
    for i in range(N_Cells):
        for j in range(N_Cells):
            depend_col.append([i, j])

    for c, element in enumerate(depend_col):
        if element[0] == element[1]:
            prob_val = 1/(N_Cells - 1)
        else:
            prob_val = 1/(N_Cells - 2)
        G_dependency[:, c] = prob_val
        for r in range(N_Cells):
            if r in element:
                G_dependency[r, c] = 0


    # node_tuple = [("S","K"),("S","G"),("K","G")]

    if env_type == "KeyCorridor" or env_type == "DoorKey":
        node_tuple = [("S","K"),("K","G")]

        model = BayesianNetwork(node_tuple)
        cpd_S = TabularCPD(variable="S", variable_card=N_Cells, values=[[1/N_Cells]]*N_Cells)
        cpd_K = TabularCPD(variable="K", variable_card=N_Cells, values=K_dependency, evidence=["S"], evidence_card=[N_Cells])
        cpd_G = TabularCPD(variable="G", variable_card=N_Cells, values=K_dependency, evidence=["K"], evidence_card=[N_Cells])
        # cpd_G = TabularCPD(variable="G", variable_card=N_Cells, values=G_dependency, evidence=["K", "S"], evidence_card=[N_Cells, N_Cells])

        model.add_cpds(cpd_S,cpd_K,cpd_G)
    elif env_type == "LavaGap":
        node_tuple = [("S","G")]

        model = BayesianNetwork(node_tuple)
        cpd_S = TabularCPD(variable="S", variable_card=N_Cells, values=[[1/N_Cells]]*N_Cells)
        cpd_G = TabularCPD(variable="G", variable_card=N_Cells, values=K_dependency, evidence=["S"], evidence_card=[N_Cells])

        model.add_cpds(cpd_S,cpd_G)
        
    model.check_model()

    return model, empty_cells

def create_bayes_network(map_lim, grid_type=6):
    node_depend = dict()
    node_tuple = []
    nodes = []
    for r in range(map_lim):
        for c in range(map_lim):
            current_grid = str(r)+"_"+str(c)
            nodes.append(current_grid)
            node_depend[current_grid] = []
            # node_tuple.append((current_grid))
            if (c-1) >= 0:
                node_tuple.append((str(r)+"_"+str(c-1), current_grid))
                node_depend[current_grid].append(str(r)+"_"+str(c-1))
            if (r-1) >= 0:
                node_tuple.append((str(r-1)+"_"+str(c), current_grid))
                node_depend[current_grid].append(str(r-1)+"_"+str(c))

    model = BayesianNetwork(node_tuple)
    for node in nodes:
        # print ("node: ", node)
        if len(node_depend[node]) == 0:
            cpd = TabularCPD(variable=node, variable_card=grid_type, values=[[0.4],[0.1],[0.1],[0.1],[0.1],[0.2]])
        elif len(node_depend[node]) == 1:
            cpd = TabularCPD(variable=node, variable_card=grid_type, values=single_depend, evidence=node_depend[node], evidence_card=[grid_type])
        elif len(node_depend[node]) == 2:
            cpd = TabularCPD(variable=node, variable_card=grid_type, values=double_depend, evidence=node_depend[node], evidence_card=[grid_type, grid_type])
        
        model.add_cpds(cpd)

    return nodes, model

def compare_elementwise(current, target):
    assert(current.shape == target.shape)
    
    diff = 0
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            if (current[r][c] != target[r][c]):
                diff += abs(target[r][c] - current[r][c])
    return diff / np.product(target.shape)


def eliminate_similar_maps(generated_maps, previous_maps):
    index_list = []
    map_list = []
    threshold = 0.25
    for index, current_map in enumerate(generated_maps):
        should_add_new = True
        for prev_map in previous_maps: # check if it is similar to the previously generated map
            comp_val = compare_elementwise(prev_map, current_map)
            if comp_val < 0.5:
                should_add_new = False

        if should_add_new:
            for added_map in map_list: # check if it is similar to the currently generated map
                comp_val = compare_elementwise(added_map, current_map)
                if comp_val < 0.5:
                    should_add_new = False
                    
        if should_add_new:
            map_list.append(current_map)
            index_list.append(index)
    
    return index_list

def create_maps(model, nodes, N_maps, map_lim, inner_map_lim):
    N_sample = 500000
    n_valid_maps = 0
    total_map_list = []
    total_data_list = []
    inference = BayesianModelSampling(model)
    while(n_valid_maps < N_maps):
        gen_data = inference.forward_sample(size=N_sample)
        for i in range(N_sample):
            valid_map = True
            key_dict = {2:0, 3:0}
            gen_map = gen_data.iloc[i].values.reshape((map_lim,map_lim))
            inner_map = gen_map[0:inner_map_lim, 0:inner_map_lim]
            for key in list(key_dict.keys()): # there can be 1 or 2 elements 
                key_dict[key] = np.sum(inner_map == key)

                if key_dict[key] == 0 or key_dict[key] > 2:
                    valid_map = False

            if np.sum(inner_map == 5) == 0 or np.sum(inner_map == 5) > 1 or np.sum(inner_map == 4) == 0 or np.sum(inner_map == 4) > 1: # there should be 1 init pos and target
                valid_map = False
            
            if valid_map and n_valid_maps < N_maps:
                n_valid_maps += 1
                total_data_list.append(gen_data.iloc[i].values)
                total_map_list.append(inner_map)
            elif n_valid_maps >= N_maps:
                break

    data_frame = pd.DataFrame(total_data_list, columns=nodes)
    return data_frame, total_map_list

def create_old_maps(infer, nodes, node_depend, N_maps, map_lim, inner_map_lim, grid_type=6):
    total_map_list = []
    for i in range(N_maps):
        current_map = np.zeros((map_lim, map_lim))
        while(True):
            key_dict = {2:0, 3:0, 4:0, 5:0}
            map_list = []
            for r in range(map_lim):
                for c in range(map_lim):
                    current_grid = str(r) + '_' + str(c)
                    depend_list = node_depend[current_grid]
                    while (True):
                        if len(depend_list) > 0:
                            evidence_list = dict()
                            for depend in depend_list:
                                evidence_list[depend] = current_map[r][c]
                            
                            # print ("prob: ", infer.query([current_grid], evidence=evidence_list).values)
                            grid_val = np.random.choice(grid_type, p=infer.query([current_grid], evidence=evidence_list).values)
                        else:
                            # print ("prob: ", infer.query([current_grid]).values)
                            grid_val = np.random.choice(grid_type, p=infer.query([current_grid]).values)


                        if grid_val in list(key_dict.keys()):
                            if key_dict[grid_val] == 0 and r < inner_map_lim and c < inner_map_lim:
                                current_map[r][c] = grid_val
                                map_list.append(grid_val)
                                key_dict[grid_val] += 1
                                break
                            else:
                                continue
                        else:
                            current_map[r][c] = grid_val
                            map_list.append(grid_val)
                            break
            if np.sum(np.array(list(key_dict.values())) > 0) == len(list(key_dict.keys())): # if key, door and goal are in a map
                break
        # print (f"map {i}: ")
        # print (np.array(map_list).reshape((map_lim, map_lim)))
        total_map_list.append(map_list)
        df_data = pd.DataFrame(total_map_list, columns = nodes)
    return df_data, np.array(total_map_list)


def create_randomized_maps(map_lim, N_maps = 100, N_sample=10000):
    prob_map = [0.5, 0.28, 0.1, 0.1, 0.01, 0.01]
    total_map_list = []
    n_valid_maps = 0
    for i in range(N_sample):
        current_map = np.zeros((map_lim, map_lim))
        for r in range(map_lim):
            for c in range(map_lim):
                current_map[r][c] = np.random.choice(np.arange(len(prob_map)), p=prob_map)
        valid_map = True
        key_dict = {2:0, 3:0}
        for key in list(key_dict.keys()): # there can be 1 or 2 elements 
            key_dict[key] = np.sum(current_map == key)
            if key_dict[key] == 0 or key_dict[key] > 2:
                valid_map = False

        if np.sum(current_map == 4) != 1 or np.sum(current_map == 5) != 1: # there should be 1 init pos and target
            valid_map = False
        
        if valid_map:
            n_valid_maps += 1
            total_map_list.append(current_map)

        if n_valid_maps >= N_maps:
            break

    return total_map_list

def obsolute_create_bayes_network(map_lim):
    nodes = ['S']
    node_depend = dict()
    node_tuple = []
    for r in range(map_lim):
        for c in range(map_lim):
            current_grid = str(r)+"_"+str(c)
            nodes.append(current_grid)
            node_depend[current_grid] = []
            node_tuple.append(('S',current_grid))
            if (c-1) >= 0:
                node_tuple.append((str(r)+"_"+str(c-1), current_grid))
                node_depend[current_grid].append(str(r)+"_"+str(c-1))
            if (r-1) >= 0:
                node_tuple.append((str(r-1)+"_"+str(c), current_grid))
                node_depend[current_grid].append(str(r-1)+"_"+str(c))

            node_depend[current_grid].append('S')
            
    model = BayesianModel(node_tuple)

    N_grids = len(nodes) - 1
    for node in nodes:
            if node == 'S':
                    cpd = TabularCPD(variable=node, variable_card=N_grids, values=[[1.0/N_grids]]*N_grids)
            else:
                    evidence_card = [6]*(len(node_depend[node])-1) + [map_lim**2] 
                    value_arr = np.tile(np.array([[0.3],[0.2],[0.2],[0.2],[0.1],[0.0]]), (1,(map_lim**2)*6**(len(node_depend[node])-1)))
                    init_grid = np.where(np.array(nodes)==node)[0][0] - 1
                    for column_update in range(init_grid, value_arr.shape[1], map_lim**2):
                            value_arr[:, column_update] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    cpd = TabularCPD(variable=node, variable_card=6, values=value_arr, evidence=node_depend[node], evidence_card=evidence_card)

            model.add_cpds(cpd)

    return model


def fix_invalid_map(map_0): 
    map_1 = np.copy(map_0)
    if np.sum(map_1 == 5) > 1: # more than one start point
        indices = np.where(map_1 == 5)
        start_point = np.random.choice(list(indices[0]))
        map_1[map_1 == 5] = 0
        map_1[start_point] = 5
    elif np.sum(map_1 == 5) == 0: # no starting point 
        indices = np.where(map_1 == 0) 
        goal_point = np.random.choice(list(indices[0])) # pick one of empty cells
        map_1[goal_point] = 5
    if np.sum(map_1 == 4) == 0: # no goal 
        indices = np.where(map_1 == 0) 
        goal_point = np.random.choice(list(indices[0])) # pick one of empty cells
        map_1[goal_point] = 4
        

    return map_1
