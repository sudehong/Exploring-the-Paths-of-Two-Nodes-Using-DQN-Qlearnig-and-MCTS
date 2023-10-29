from graph_editor import GraphEditor, Reaction, GraphRewrite, VectorCompare,check_all_action
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from collections import defaultdict, deque, namedtuple
from scipy.sparse import coo_matrix, save_npz, load_npz,lil_matrix
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from copy import deepcopy
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# def load_data2(path):
#   graph_edit = GraphEditor()
#   graph_data = []
#   with open(path) as f:
#     l = f.readline()
#     while l:
#       reaction_id, rxn = l.strip().replace("Fr", "H").split()
#       reactant, product = graph_edit(rxn)
#       if isinstance(reactant, type(None)):
#           pass
#       elif isinstance(product, type(None)):
#           pass
#       else:
#         graph_data.append([rxn,reactant,product, reaction_id])
#       l = f.readline()
#   random.seed(0)
#   return graph_data
#
#
# paths = "output.txt"
# graph_data = load_data2(paths)
#
# print(graph_data)




'''
    create Pss_matrix from (graph_editor.check_all_action),the order of all state is stored in smiles_list
'''


# rxn_smi = "BrC1=CC=CC=C1.OB(O)C2=CC=CC=C2.[Pd]>>C3=CC=C(C=C3)C4=CC=CC=C4.[Pd].OB(O)Br" #49
# rxn_smi = "BrC1=CC=CC=C1.CCCC[Sn](CCCC)(CCCC)C2=CC=CC=C2.[Pd]>>Br[Sn](CCCC)(CCCC)CCCC.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"
# rxn_smi = "BrC1=CC=CC=C1.Br[Mg]C2=CC=CC=C2.[Pd]>>Br[Mg]Br.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"

# rxn_smi = "[CH3:2][O:1][C:3]1=[CH:4][CH:6]=[C:8](Cl)[CH:7]=[CH:5]1.OB([C:9]2=[CH:10][CH:12]=[CH:14][CH:13]=[CH:11]2)O.[Pd:15]>>[CH3:2][O:1][C:3]3=[CH:4][CH:6]=[C:8]([C:9]4=[CH:10][CH:12]=[CH:14][CH:13]=[CH:11]4)[CH:7]=[CH:5]3.[Pd:15]"
# rxn_smi = "BrC1=CC=C(C2=CC=C(C)C=C2)C=C1.OB(C3=CC=CC=C3)O.[Pd]>>CC4=CC=C(C5=CC=C(C=C5)C6=CC=CC=C6)C=C4.OB(O)Br.[Pd]"
# rxn_smi = "[CH3:2][C:1]([C:4]1=[CH:5][CH:7]=[C:9](Cl)[CH:8]=[CH:6]1)=[O:3].OB([C:10]2=[CH:11][CH:13]=[CH:15][CH:14]=[CH:12]2)O.[Pd:16]>>[CH3:2][C:1]([C:4]3=[CH:5][CH:7]=[C:9]([C:10]4=[CH:11][CH:13]=[CH:15][CH:14]=[CH:12]4)[CH:8]=[CH:6]3)=[O:3].[Pd:16]"
# rxn_smi = "OB(O)C1=CC=CC=C1.BrC2=CC=C(C)C=C2.[Pd]>>CC(C=C3)=CC=C3C4=CC=CC=C4.[Pd]"
# rxn_smi = "CC(O1)(C)C(C)(C)OB1C2=CC=CC=C2.BrC3=CC=C(C)C=C3.[Pd]>>CC(C=C4)=CC=C4C5=CC=CC=C5.[Pd]"
rxn_smi = "BrC1=CC=C(Br)C=C1.OB(O)C2=CC=C(C)C=C2.[Pd]>>BrC3=CC=C(C=C3)C4=CC=C(C)C=C4.[Pd].OB(Br)O"

file = check_all_action(rxn_smi)
smiles_list = file[0]
action_mask = file[2]



# for i in tqdm(range(len(graph_data))):
#     file = check_all_action(graph_data[i][0])
#     smiles_list = file[0]
#     if len(smiles_list)>=300:
#         continue
#     else:
#         with open('output.txt', 'a') as file:
#             file.write(graph_data[i][0] + '\n')

Pss_matrix1 = file[1]

Q_table = lil_matrix((Pss_matrix1.shape[0], Pss_matrix1.shape[1]), dtype=float)


'''
    Because the molecules obtained by multiple smile writing methods may be the same molecule, they are converted into graphs to determine and use the graphs to find the matrix index of reactants and prod
'''
def find_value(dictionary, target):
  for key, value in dictionary.items():
    if key == target:
      return value
  return None

def find_key(dictionary, target):
  for key, value in dictionary.items():
    if value == target:
      return key
  return None

print(find_key(smiles_list, 7))
print(find_key(smiles_list, 44))
print(find_key(smiles_list, 69))
print(find_key(smiles_list, 72))
print(find_key(smiles_list, 78))


graph_edit = GraphEditor()
pyg_react, pyg_prod = graph_edit(rxn_smi)
react_smiles = graph_edit.get_smiles(pyg_react)
prod_smiles = graph_edit.get_smiles(pyg_prod)
keys_list = list(smiles_list.keys())
print(len(keys_list))

pyg_graph = [graph_edit.smiles2graph(i) for i in keys_list]
# print('=======')
# print(smiles_list)
react_index = find_value(smiles_list, react_smiles)
prod_index = find_value(smiles_list, prod_smiles)

compare = VectorCompare(pyg_prod)

for key ,value in smiles_list.items():
    t = graph_edit.smiles2graph(key)
    if compare(t):
        prod_index = value

print(react_index,prod_index)

'''
    Use matrix to Q-learning
'''
epsilon = 0.9
for episode in tqdm(range(1000)):
    # print("episode", episode + 1)
    # if episode != 0:
    #     epsilon *= 0.99
    state = react_index
    target_state = prod_index
    ntry = 100
    while 1:
        actions = Pss_matrix1[state, :].nonzero()[1].tolist()
        if np.random.rand() < epsilon:
            next_state = np.random.choice(actions)
        else:
            qs = [Q_table[state, a] for a in actions]
            next_state = actions[np.argmax(qs)]
        if target_state == next_state:
            done = True
            reward = 1
            next_q_max = 0
        else:
            done = False
            reward = 0
            next_qs = [Q_table[next_state, a] for a in Pss_matrix1[next_state, :].nonzero()[1].tolist()]
            next_q_max = max(next_qs)
        # print("state next_state, reward, done", state, next_state, reward, done)

        target = reward + 0.9 * next_q_max
        Q_table[state, next_state] += (target - Q_table[state, next_state]) * 0.8
        # state = next_state
        if done:
            break
        else:
            state = next_state
        ntry -= 1

print(Q_table)

'''
    Using the obtained Q-value table and the dfs algorithm, identify all paths, which may have multiple paths
'''
state = react_index
target_state = prod_index

def find_max_indices(nums):
    if len(nums) == 0:
        return []
    max_value = max(nums)
    indices = [i for i, num in enumerate(nums) if num == max_value]
    return indices

maze = dict()

for states in range(len(pyg_graph)):
    actions = Pss_matrix1[states, :].nonzero()[1].tolist()
    qs = [Q_table[states, a] for a in actions]
    qq = list(qs)
    #有多个路径
    max_index_list = find_max_indices(qq)
    l = []
    for i in max_index_list:
        col = actions[i]
        if col in l:
            pass
        else:
            l.append(col)
    maze[states] = l

print(maze)

def find_all_paths(graph, start, end):
    paths = []  # 用于存储所有路径的列表
    queue = deque([(start, [start])])  # 使用队列来进行广度优先搜索

    while queue:
        current_node, path = queue.popleft()
        if current_node == end:  # 如果当前节点是终点，将路径添加到结果列表中
            paths.append(list(path))
        else:
            for neighbor in graph[current_node]:
                if neighbor not in path:  # 仅考虑未访问过的邻居节点
                    queue.append((neighbor, path + [neighbor]))

    return paths


start_state = state
end_state = target_state

paths = find_all_paths(maze, start_state, end_state)

print(paths)
print("All paths from", start_state, "to", end_state, ":")


for i in range(len(paths)):
    path = paths[i]
    s = [find_key(smiles_list,ii) for ii in path]
    print(s)
    mol_list = [Chem.MolFromSmiles(j) for j in s]
    img = Draw.MolsToGridImage(mol_list, molsPerRow=len(mol_list))
    # 生成动态图像名称
    image_name = "grid_image_{}.png".format(i)
    # 保存图像到文件
    img.save(image_name)

    print(path)



