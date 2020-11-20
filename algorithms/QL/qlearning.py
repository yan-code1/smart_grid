import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, epsilon=0.1, alpha=0.1, gamma=0.9, q_init=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.qtable = np.ones((n_states, n_actions)) * q_init
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_init = q_init

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        max_actions = np.where(self.qtable[state] == self.qtable[state].max())[0]
        action = np.random.choice(max_actions)
        return action
    
    def learn(self, state, state_next, action, reward):
        q_next = self.qtable[state_next].max()
        q_target = reward + self.gamma * q_next
        q_eval = self.qtable[state, action]
        self.qtable[state, action] = q_eval * (1 - self.alpha) + q_target * self.alpha

    def reset(self):
        self.qtable = np.ones((self.n_states, self.n_actions)) * self.q_init

    def get_state_q(self, state):
        return self.qtable[state]

    def save_q(self, path):
        import scipy.io as sio
        sio.savemat(path, {"Q": self.qtable})

    def load_q(self, path):
        import scipy.io as sio
        self.qtable = sio.loadmat(path)["Q"]

        
