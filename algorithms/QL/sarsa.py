import numpy as np
import config.env_robot_config as ec
import config.global_exp_config as gc
class sarsa:
    def __init__(self,state_num = ec.STATE_NUM, action_num = ec.ACTION_NUM ,alpha = 0.1,gamma=0.9,epsilon_start=0.1,epsilon_end=0.01):
        self.state_num = state_num
        self.action_num = action_num
        # self.num_iter = num_iter
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        # iter_num = []
        self.qvalue = np.zeros((state_num,action_num))




    def reset(self):
        self.qvalue = np.zeros((self.state_num,self.action_num))
        # self.num_iter = []
        self.anneal_step = (self.epsilon_start - self.epsilon_end) / (gc.STEP_NUMBER - 2)
    def epsilon_anneal(self):
        self.epsilon -= self.anneal_step
        if self.epsilon < 1e-6:  # EPS
            self.epsilon = 0
    def choose_action(self,state):
        state_idx = ec.state_result_encode(state)
        self.epsilon_anneal()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_num)
        max_actions = np.where(self.qvalue[state_idx] == self.qvalue[state_idx].max())[0]
        action = np.random.choice(max_actions)
        return action

    def learn(self,state,state_next,action,action_next):
        state_idx = ec.state_result_encode(state)
        state_idx_next = ec.state_result_encode(state_next)
        reward = ec.reward_function(state,state_next)
        q_next = self.qvalue[state_idx_next,action_next]
        q_target = reward + q_next*self.gamma
        q_eval = self.qvalue[state_idx,action]
        self.qvalue[state_idx,action] =  self.alpha * (q_target-q_eval)+q_eval

