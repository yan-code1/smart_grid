import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time
from config.utils  import _get_order_array,fourier_basis
class ActorCritic:
    def __init__(self, weight_init, num_actions, gamma, alpha, beta1, epsilon, epsilon_decay, epsilon_min,
                 action_selection_method="e-greedy", temperature=None, func_approx_type='fourier', order=None,
                 num_state_dimensions=None, num_states=None):
        self.weight_init = weight_init
        self.order = order
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta1
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_state_dimensions = num_state_dimensions
        self.func_approx_type = func_approx_type
        self.order = order

        self.weight_init = weight_init
        self.action_selection_method = action_selection_method
        self.temperature = temperature
        self.epsilon = self.epsilon_start

        self.order_list = _get_order_array(self.order, self.num_state_dimensions, start=0)

        self.reset()


    def reset(self):
        #生成随机矩阵
        self.theta = self.weight_init*np.random.randn(self.num_actions, self.num_state_dimensions)
        self.w =     self.weight_init*np.random.randn(len(self.order_list))

    #TD——ERROR
    def get_value_TD_error(self, r, state, state_prime):
        # start1 = time.time()
        pred = self.value_approximate(state_prime) #先前状态——V值计算
        td = r + self.gamma * pred - self.value_approximate(state)
        # end = time.time()
        # print("time_get:", end - start1)
        return td

    def update_actor(self, state, action, td_error):
        row = self.action_value_approximate(state)
        probs = self.softmax(row)
        pi_s = probs
        # b = np.zeros((self.num_actions, len(self.order_list)))
        # for i in range(self.num_actions):
        #     if i == action:
        #         b[i] = (1 - pi_s[i]) * self.phi(state).reshape(1, -1)
        #     else:
        #         b[i] = (-pi_s[i]) * self.phi(state).reshape(1, -1)
        b = (-pi_s * np.array(state).reshape(-1,1)).reshape(self.num_actions, self.num_state_dimensions)
        b[action] = (1 - pi_s[action]) * np.array(state).reshape(1, -1)
        self.theta += self.beta*td_error*b

    def update_critic(self,state,action, td_error):
        self.w = self.w + self.alpha*(td_error) * self.phi(state).reshape(1,-1)


    def phi(self,state):
        return fourier_basis(state, self.order_list)
    #V值计算函数
    def value_approximate(self,state):

        np.dot(self.w, self.phi(state))

        return np.dot(self.w, self.phi(state))

    def action_value_approximate(self,state, action = None):
        '''
        Computes Q(s,a) using Function Approximation
        Returns one approximation for each action if action is None
        If there is an action, returns the Q(s,a) for that action
        '''

        Q_s = np.dot(self.theta, np.array(state).reshape(-1,1))
        assert Q_s.shape == (self.num_actions, 1)
        if action == None:
            return Q_s
        else:
            return Q_s[action]

    def select_action(self,state):
        row = self.action_value_approximate(state)

        # print(row)
        probs = self.softmax(row)
        # print(probs)
        return int(np.random.choice(self.num_actions, 1, p=probs))

    def softmax(self, row):
        # probs = (np.exp((1 / self.epsilon) * row) / np.sum(np.exp((1 / self.epsilon) * row)))
        probs = (np.exp((1 / self.epsilon) * (row-np.max(row))) / np.sum(np.exp((1 / self.epsilon) * (row-np.max(row)))))
        probs = probs.reshape(-1)
        return probs


