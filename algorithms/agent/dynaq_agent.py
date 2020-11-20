import numpy as np

import random
from algorithms.QL.qlearning import QLearning
import config.env_robot_config as ec

class DynaQAgent:
    def __init__(self,dyna_num):
        self.ql = QLearning(ec.STATE_NUM, ec.ACTION_NUM)
        self.experiences=[]
        self.dyna_num=dyna_num
    def reset(self):
        self.ql.reset()

    def choose_action(self, state_results, epsilon=None):
        state_idx = ec.state_result_encode(state_results)
        return self.ql.choose_action(state_idx, epsilon=epsilon)

    def learn(self, state_results, state_results_next, action_idx):
        state_idx = ec.state_result_encode(state_results)
        state_next_idx = ec.state_result_encode(state_results_next)
        reward = ec.reward_function(state_results,state_results_next)
        self.ql.learn(state_idx, state_next_idx, action_idx, reward)
        self.experiences.append([state_idx, state_next_idx, action_idx, reward])
        for  i in range(self.dyna_num):
            a=self.experiences
            experience=random.choice(self.experiences)
            state_ex = experience[0]
            state_next_ex = experience[1]
            action_ex = experience[2]
            reward_ex = experience[3]
            self.ql.learn(state_ex, state_next_ex, action_ex, reward_ex)

