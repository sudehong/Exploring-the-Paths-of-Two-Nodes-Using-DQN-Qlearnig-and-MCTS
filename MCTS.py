import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.nnconv1 import NNModel
import copy
from collections import namedtuple,deque
import random
import numpy as np
import torch.nn as nn
from graph_editor import GraphEditor, Reaction, GraphRewrite, VectorCompare,check_all_action
import networkx as nx
import torch.optim as optim
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from torch_geometric.data import Batch
import math
from scipy.sparse import lil_matrix,coo_matrix
import gc


# rxn_smi = "BrC1=CC=CC=C1.OB(O)C2=CC=CC=C2.[Pd]>>C3=CC=C(C=C3)C4=CC=CC=C4.[Pd].OB(O)Br"
# rxn_smi = "BrC1=CC=CC=C1.Br[Mg]C2=CC=CC=C2.[Pd]>>Br[Mg]Br.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"

# rxn_smi = "BrC1=CC=CC=C1.CCCC[Sn](CCCC)(CCCC)C2=CC=CC=C2.[Pd]>>Br[Sn](CCCC)(CCCC)CCCC.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"
# rxn_smi = "CC(O1)(C)C(C)(C)OB1C2=CC=CC=C2.BrC3=CC=C(C)C=C3.[Pd]>>CC(C=C4)=CC=C4C5=CC=CC=C5.[Pd]"
# rxn_smi ="CCCC[Sn](CCCC)([C:1]1=[CH:2][CH:4]=[CH:6][CH:5]=[CH:3]1)CCCC.Cl[C:7]2=[CH:8][CH:10]=[CH:12][CH:11]=[CH:9]2.[Pd:13]>>[C:7]3([C:1]4=[CH:2][CH:4]=[CH:6][CH:5]=[CH:3]4)=[CH:8][CH:10]=[CH:12][CH:11]=[CH:9]3.[Pd:13]"
# rxn_smi = "BrC1=CC=C(Br)C=C1.OB(O)C2=CC=C(C)C=C2.[Pd]>>BrC3=CC=C(C=C3)C4=CC=C(C)C=C4.[Pd].OB(Br)O"

def load_data(path):

  graph_data = []
  with open(path) as f:
    l = f.readline()
    while l:
      reaction_id, rxn = l.strip().replace("Fr", "H").split()
      graph_data.append(rxn)
      l = f.readline()

  return graph_data

def find_value(dictionary, target):
  for key, value in dictionary.items():
    if key == target:
      return value
  return None

#state是smile分子式
#根据矩阵找出此时state下的能转移到的next_state的graph
def find_next_state(state):
    state_index = find_value(smiles_list, state)
    actions = Pss_matrix1[state_index, :].nonzero()[1].tolist()
    next_state_smile = [keys_list[i] for i in actions]
    return next_state_smile

def find_next_state_graph(state):
    state_index = find_value(smiles_list, state)
    actions = Pss_matrix1[state_index, :].nonzero()[1].tolist()
    next_state_smile = [pyg_graph[i] for i in actions]
    return next_state_smile



#playout
def playout(state):
    #往后走8步,遇到目标就停止输出1，8步内没遇到就输出0
    state_graph = graph_edit.smiles2graph(state)
    if compare(state_graph):
        return 1
    ntry = 8
    while (ntry>0):
        state_index = find_value(smiles_list, state)
        actions = Pss_matrix1[state_index, :].nonzero()[1].tolist()
        # print(state_index)
        # print(actions)

        #如果碰到叶子节点
        if not actions:
            return 0
        next_state_index = np.random.choice(actions)
        next_graph = pyg_graph[next_state_index]
        next_smile = keys_list[next_state_index]
        if compare(next_graph):
            # print(111)
            return 1
        state = next_smile
    if ntry == 0 :
        return -1

def mcts_action(state):
    #定义mcts的node
    class Node:
        def __init__(self, state):
            self.state = state
            self.w = 0  # 価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード

        def evaluate(self):
            # 探索完的时候
            state_graph = graph_edit.smiles2graph(self.state)
            if compare(state_graph):
                # 从胜败结果中获取价值
                value = 1 if compare(state_graph) else 0

                # 更新累计价值和试行回数
                self.w += value
                self.n += 1
                return value

            # 如果不存在子节点时
            if not self.child_nodes:
                # 从神经网络中找出方策和价值
                # policies, value = predict(model, self.state)
                value = playout(self.state)
                # print(value)

                # 更新累计价值和试行回数
                self.w += value
                self.n += 1

                # 展开节点
                # if self.n == 10:
                self.expand()
                return value
            # 节点存在的时候
            else:
                value = self.next_child_node().evaluate()
                self.w += value
                self.n += 1
                return value

        # 展开节点
        def expand(self):
            self.child_nodes = []
            actions = find_next_state(self.state)
            for s in actions:
                self.child_nodes.append(Node(s))


        def next_child_node(self):
            # 返回n为零的子节点
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node
            # UCB1的计算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(child_node.w / child_node.n + 2*(math.log(t) / child_node.n) ** 0.5)

            # 返回UCB1最大的节点
            # print(ucb1_values)
            return self.child_nodes[np.argmax(ucb1_values)]

    #生成现在局面的node
    root_node = Node(state)
    root_node.expand()

    #100回模拟
    for _ in tqdm(range(2000)):
        root_node.evaluate()

    #返回n最大的行动
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    # print(n_list)
    legal_actions = find_next_state(state)
    return legal_actions[np.argmax(n_list)]

def random_action(state):
    legal_actions = find_next_state(self.state)
    return legal_actions[random.randint(0,len(legal_actions)-1)]

class State():
    def __init__(self, s):
        self.state = s



def play(next_action):

    env = State(pyg_react)
    ntry = 8
    state = env.state
    state_smiles = graph_edit.get_smiles(state)
    path = []
    path.append(state_smiles)
    while (ntry>0):
        state_smiles = graph_edit.get_smiles(state)
        #获取行动
        # print("--")
        # print(smiles_list[state_smiles])
        # print(state_smiles)
        #判断下一状态中是否有目标
        next_list = find_next_state_graph(state_smiles)

        flag = 0
        for i in next_list:
            if compare(i):
                path.append(graph_edit.get_smiles(i))
                # print(graph_edit.get_smiles(i))
                print("I win")
                flag = 1
                break
        if flag:
            break

        next_smile = next_action(state_smiles)
        # print(f"next smile :{next_smile}")
        # print("--------")
        path.append(next_smile)
        next_graph = graph_edit.smiles2graph(next_smile)
        state = next_graph
        ntry -= 1

    # print(path)
    return path

if __name__ == "__main__":

    path = "Stille_coupling.txt"

    # path = "Suzuki_Miyaura_coupling.txt"
    smi_list = load_data(path)
    # print(smi_list)
    # print(len(load_data(path)))
    # smi_list1 = smi_list[21:]
    # smi_list1 = ["CCCC[Sn](CCCC)([C:1]1=[CH:2][CH:4]=[CH:6][CH:5]=[CH:3]1)CCCC.[O:8]=[N:7]([C:10]2=[CH:11][CH:13]=[C:15](Cl)[CH:14]=[CH:12]2)=[O:9].[Pd:16]>>[O:8]=[N:7]([C:10]3=[CH:11][CH:13]=[C:15]([C:1]4=[CH:2][CH:4]=[CH:6][CH:5]=[CH:3]4)[CH:14]=[CH:12]3)=[O:9].[Pd:16]",
    #              "BrC1=CC=CC=C1.Br[Mg]C2=CC=CC=C2.[Pd]>>Br[Mg]Br.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"]
    k = 0

    for index,item in enumerate(tqdm(smi_list)):
        if index < 24:
            continue

        rxn_smi = item
        # print(index)
        print(rxn_smi)
        file = check_all_action(rxn_smi)
        smiles_list = file[0]
        Pss_matrix1 = file[1]

        # #从反应物到生成物
        device = torch.device("cpu")
        graph_edit = GraphEditor()
        pyg_react, pyg_prod = graph_edit(rxn_smi)

        react_smiles = graph_edit.get_smiles(pyg_react)

        rewrite = GraphRewrite(device)
        compare = VectorCompare(pyg_prod)
        keys_list = list(smiles_list.keys())

        next_action = mcts_action
        try:
            pyg_graph = [graph_edit.smiles2graph(i) for i in keys_list]
            reaction_path = play(next_action)
            print(reaction_path)
            combined_text = f"{index} {item} {reaction_path}"
            with open('output.txt', 'a') as file:
                file.write(combined_text+ "\n")

        except Exception as e:
            k += 1
            continue

        print(f"异常smile数量:{k}")
        gc.collect()
    print(f"异常smile数量:{k}")
