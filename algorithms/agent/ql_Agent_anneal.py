
import config.ql_anneal_config as qc
import config.global_exp_config as gc
import config.comm_config as cc
import numpy as np
from helper.data_move_saver import DataSaver
# from utils.data_saver import DataSaver
from algorithms.QL.qlearning import QLearning
import random
import math

class QL_Agent:
    def __init__(self,STATE_NUM,ACTION_NUM,epsilon_start=1, epsilon_end=0.01,decay = 200):
        self.ql= QLearning(STATE_NUM,ACTION_NUM)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.decay = decay
        self.anneal_rate = (epsilon_start - epsilon_end) / gc.STEP_NUMBER
        self.step_count = 0

    def epsilon_anneal(self):

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (math.exp(-1.0 * self.step_count / self.decay))  # 保持起点一致

        # self.epsilon -= self.anneal_step
        # if self.epsilon < 1e-6: # EPS
        #     self.epsilon = 0

    def reset(self):
        self.ql.reset()
        self.epsilon = self.epsilon_start
        self.step_count = 0
        # self.anneal_step = (self.epsilon_start - self.epsilon_end) / (gc.STEP_NUMBER - 2)


    def choose_action(self, state_idx,epsilon = None):
        if epsilon == None:
           epsilon = self.epsilon
           self.epsilon_anneal()

        action=self.ql.choose_action(state_idx,epsilon)
        return action


    def learn(self, state_idx, state_next_idx, action,reward):
        self.step_count += 1
        self.ql.learn(state_idx,state_next_idx,action,reward)


