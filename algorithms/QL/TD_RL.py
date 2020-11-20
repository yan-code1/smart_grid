import numpy as np
from comm_env.env_move import *

class TD_RL:
    def __init__(self,gamma = 0.8):
        self.gamma =gamma
        self.qvalue


        def greedy_policy(self,qfun,state):
            action_max = qfun[state,:].argmax()
            return