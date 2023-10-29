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
# from models.dnnconv import NNModel



Action = namedtuple('Action', 'type u v')


class ReplayBuffer(object):
  def __init__(self, mem_size, batch_size):
    self.buffer = deque(maxlen=mem_size)
    self.batch_size = batch_size

  def __len__(self):
      return len(self.buffer)

  def store_transition(self, state, actidx, reward, new_state, done):
    data = (state, actidx, reward, new_state, done)
    self.buffer.append(data)

#  returns batch of samples
  def sample_memory(self):
    data = random.sample(self.buffer, self.batch_size)
    # print(len(self.buffer))
    # data = self.buffer
    state_list = [x[0] for x in data]
    actidxs = [x[1] for x in data]
    rewards = [x[2] for x in data]
    new_state_list = [x[3] for x in data]
    dones = [x[4] for x in data]
    return state_list, actidxs, rewards, new_state_list, dones

  def all_data(self): 
    data = self.buffer
    state_list = [x[0] for x in data]
    actidxs = [x[1] for x in data]
    rewards = [x[2] for x in data]
    new_state_list = [x[3] for x in data]
    dones = [x[4] for x in data]
    return state_list, actidxs, rewards, new_state_list, dones
      


device = torch.device("cpu")

class Agent:
  def __init__(self, gnn, loss_fnc, optimizer,compare, epsilon=1.0, gamma=0.9, r=0.99, lr=0.01, batch_size=128, mem_size=1000):
    self.q = gnn
    # self.q2 = gnn2
    # self.q2.load_state_dict(self.q.state_dict())
    # self.q2.eval()
    self.loss_fnc = loss_fnc  # 誤差関数
    self.optimizer = optimizer  # 最適化アルゴリズム
    self.epsilon = epsilon  # ε
    self.gamma = gamma  # 割引率
    self.r = r  # εの減衰率
    self.lr = lr  # 学習係数
    self.batch_size = batch_size  #  or global params?
    self.replayBuffer = ReplayBuffer(mem_size=mem_size, batch_size=batch_size)
    self.episode_count = 0
    # 创建 ReduceLROnPlateau 调度器，将其关联到优化器
    self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    self.compare = compare

  def learn(self, state, action_index, reward, new_state, done):  # ニューラルネットワークを訓練
    self.replayBuffer.store_transition(state, action_index, reward, new_state, done)
    if len(self.replayBuffer) < self.batch_size:
        return 0
    # states, actidxs, rewards, new_states, dones = self.replayBuffer.sample_memory()
    # states, actidxs, rewards, new_states, dones = self.replayBuffer.all_data()
    # d = self.replayBuffer.all_data()
    # print(len(d[3]))
    # # 检查有到最后一步的数据
    # k = 0
    # for i in d[3]:
    #     if self.compare(i):
    #         k +=1
    #         print("datagoaltrue")
    #         print(k)
    #     else:
    #         pass

    states, actidxs, rewards, new_states, dones = self.replayBuffer.sample_memory()
    # self.q.eval()  # 評価モード
    targets_q = self.q(Batch.from_data_list(new_states).to(device))#[0]
    self.q.train()  # 訓練モード
    predictions = self.q.forward(Batch.from_data_list(states).to(device))#[0]

    self.optimizer.zero_grad()
    loss = torch.Tensor([0]).to(device)

    for tq, p, r, done, actidx in zip(targets_q, predictions, rewards, dones, actidxs):
      target_action = torch.argmax(tq)
      target = torch.tensor(r).to(device) + torch.mul(torch.mul(tq[target_action], self.gamma), (1 - done))
      prediction = p[actidx]
      loss += self.loss_fnc(prediction, target)

    loss.backward()
    self.optimizer.step()
    
    return loss.detach()

  def choose_action(self, state, len_actions, episode,mask):

      # epsilon の計算

      # if np.random.rand() < self.epsilon:  # ランダムな行動
      if np.random.rand() < 0.5:  # ランダムな行動
          non_zero_indices = [index for index, element in enumerate(mask) if element != 0]
          # print(non_zero_indices)
          if non_zero_indices:
              action_index = np.random.choice(non_zero_indices)
          else:
              action_index = np.random.randint(0, len_actions)

      else:
          estimates = self.q(state.to(device))
          data = estimates[0].detach().numpy()
          m = np.array(mask)
          result = data * m
          # 找到非零值的索引和值
          non_zero_indices = np.where(result != 0)[0]  # 找到非零值的索引
          non_zero_values = data[non_zero_indices]  # 找到非零值的值
          # 找到非零值中的最大值和其索引
          max_non_zero_index = np.argmax(non_zero_values)
          action_index = non_zero_indices[max_non_zero_index]

      # if self.epsilon > 0.1 and episode >= self.episode_count:  # εの下限
      #     self.epsilon *= self.r
      #     self.episode_count += 1

      return action_index

  def sync_qnet(self):
    self.q2.load_state_dict(self.q.state_dict())
        
class Environment:
    def __init__(self, pyg_react, pyg_prod):
        self.compare = VectorCompare(pyg_prod)
        self.rewrite = GraphRewrite(torch.device("cpu"))
        self.state = pyg_react
        self.actions = self.get_actions(pyg_react)  # initial values of state,actions

    def get_actions(self, graph):
        addition = [Action(Reaction.Addition, u, v) for u, v in graph.loop_edge]
        elimination = [Action(Reaction.Elimination, u, v) for u, v in graph.loop_pair]
        goal = [Action(Reaction.Goal, u, v) for u, v in graph.goal]
        actions = addition + elimination + goal
        return actions

    def step(self, action_index,ntry):
        action = self.actions[action_index]
        reaction, u, v = action
        new_state = self.rewrite(self.state, action)  # next_stateはおそらくgraph

        if self.compare(new_state) and ntry>1:
            reward = 1
            terminal = True
        elif ntry == 1:
            reward = -1
            terminal = False
        else:
            reward = 0
            terminal = False
            
        return new_state, reward, terminal

def main(cp):
    
    file = check_all_action(rxn_smi)
    smiles_list = file[0]
    action_mask = file[2]
    gnn = NNModel(in_channels=104, edge_channels=7, hidden_channels=16).to(device)
     
    loss_fnc = nn.MSELoss()  # 誤差関数
    optimizer = optim.Adadelta(gnn.parameters(), lr=0.5)
    agent = Agent(gnn, loss_fnc, optimizer, compare = cp, epsilon=0.9, batch_size=128, mem_size=10000)
    

    graph_edit = GraphEditor()
    reactant, product = graph_edit(rxn_smi)
    
    for episode in tqdm(range(300)):
        
        episode_reward = 0
        episode_loss = 0
            
        env = Environment(reactant, product)
        ntry = 8
        info = False
        while (ntry>0):  # loop over trials
            
            state = env.state
            actions = env.actions

            react_smiles = graph_edit.get_smiles(state)

            if react_smiles in action_mask:
                mask = action_mask.get(react_smiles)
            else:
                mask = [1]*len(actions)

            if all(element == 0 for element in mask ):
                break

            action_index = agent.choose_action(state, len(actions), episode,mask)
            new_state, reward, done = env.step(action_index,ntry)
            episode_reward += reward
            loss = agent.learn(state, action_index, reward, new_state, done)
            episode_loss += loss

            if done:
                print("done_true")
                break
            else:
                env = Environment(new_state, product)
            ntry -= 1
            # print(ntry)
        
        # print(episode_reward)
        print(episode_loss)
        # agent.scheduler.step(episode_loss)
            
    torch.save(gnn.state_dict(), path)
    
    
if __name__ == "__main__":
    device = torch.device("cpu")
    test = False
    path = 'ep249.pth'
    
    # rxn_smi = "BrC1=CC=CC=C1.OB(O)C2=CC=CC=C2.[Pd]>>C3=CC=C(C=C3)C4=CC=CC=C4.[Pd].OB(O)Br"
    #['Brc1ccccc1.OB(O)c1ccccc1.[Pd]', 'Brc1ccccc1.OB(O)[Pd]c1ccccc1', 'OB(O)[Pd](Br)(c1ccccc1)c1ccccc1', 'OB(O)Br.c1ccc([Pd]c2ccccc2)cc1', 'OB(O)Br.[Pd].c1ccc(-c2ccccc2)cc1']
    # rxn_smi = "Br[C:2]([H:5])([H:7])[C:1]([H:4])([H])[H:3].[Pd:6]>>[H:3]/[C:1]([H:4])=[C:2]([H:5])/[H:7].[Pd:6]"
    # rxn_smi = "[CH3:2][C:1]([C:4]1=[CH:5][CH:7]=[C:9](Cl)[CH:8]=[CH:6]1)=[O:3].OB([C:10]2=[CH:11][CH:13]=[CH:15][CH:14]=[CH:12]2)O.[Pd:16]>>[CH3:2][C:1]([C:4]3=[CH:5][CH:7]=[C:9]([C:10]4=[CH:11][CH:13]=[CH:15][CH:14]=[CH:12]4)[CH:8]=[CH:6]3)=[O:3].[Pd:16]"

    rxn_smi = "BrC1=CC=CC=C1.CCCC[Sn](CCCC)(CCCC)C2=CC=CC=C2.[Pd]>>Br[Sn](CCCC)(CCCC)CCCC.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"
    # rxn_smi = "BrC1=CC=CC=C1.Br[Mg]C2=CC=CC=C2.[Pd]>>Br[Mg]Br.C3(C4=CC=CC=C4)=CC=CC=C3.[Pd]"
    
    # rxn_smi = "OB([C:1]1=[CH:2][CH:4]=[C:6]([CH3:7])[CH:5]=[CH:3]1)O.[N:9]#[C:8][C:10]2=[C:11](Cl)[CH:13]=[CH:15][CH:14]=[CH:12]2.[Pd:16]>>[N:9]#[C:8][C:10]3=[C:11]([C:1]4=[CH:2][CH:4]=[C:6]([CH3:7])[CH:5]=[CH:3]4)[CH:13]=[CH:15][CH:14]=[CH:12]3.[Pd:16]"
    # rxn_smi = "BrC1=CC=C(C)C=C1.C2(B3OCCCO3)=CC=CC=C2.[Pd]>>CC(C=C4)=CC=C4C5=CC=CC=C5.[Pd]"
    
    #从反应物到生成物
    graph_edit = GraphEditor()
    pyg_react, pyg_prod = graph_edit(rxn_smi)
    rewrite = GraphRewrite(device)
    compare = VectorCompare(pyg_prod)
    rewrite = GraphRewrite(device)
    compare = VectorCompare(pyg_prod)
    
    main(compare)
    
    gnn1 = NNModel(in_channels=104, edge_channels=7, hidden_channels=16).to(device)
    gnn1.load_state_dict(torch.load(path))
    ntry = 8

    path = []
    path.append(graph_edit.get_smiles(pyg_react))
    state = pyg_react

    print("start")
    while (ntry>0):
        qs = gnn1(state)[0].tolist()
        max_index = np.argmax(qs)
        action = graph_edit.get_actions(state)

        nx_prod = rewrite(state, action[max_index])
        smile = graph_edit.get_smiles(nx_prod)

        if compare(nx_prod):
            print("true")
            path.append(smile)
            break
        state = nx_prod
        path.append(smile)
        if ntry == 1:
            path.clear()
        ntry -= 1

    print(path)

