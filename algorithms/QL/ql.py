import random
import numpy as np
import scipy.io as iso


class QLearning():
    def __init__(self,state_num,action_num,alpha=0.1,epislon=0.1,gamma=0.8):
        self.state_num = state_num
        self.action_num = action_num
        self.alpha = alpha
        self.epislon = epislon
        self.gamma = gamma
        self.qtable = np.zeros((self.state_num,self.action_num))
        self.ql_init = 0


    def choose_action(self,state,epislon = None):
        if epislon == None:
            epislon = self.epislon
        if epislon < np.random.rand():
            action = np.random.choice(self.action_num)
        else:
            action_max = np.where(self.qtable[state] == self.qtable[state].max())[0]
            action = random.choice(action_max)
        return action

    def learn(self,state,state_next,action,reward):
        ql_next = self.qtable[state_next].max()
        ql_target = reward + ql_next*self.gamma

        ql_eval = self.qtable[state,action]
        self.qtable[state, action] = ql_target*self.alpha + ql_eval*(1-self.alpha)

    def reset(self):
        self.qtable = np.zeros((self.state_num,self.action_num))

    def save_q(self,path):
        iso.savemat(path,{"Q": self.qtable})
