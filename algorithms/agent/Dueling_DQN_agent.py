import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, random
import config.env_robot_config as ec

class Due_DQNAgent:
    def __init__(self,**kwargs):
        self.due_dqnagent = DuelingDQN(**kwargs)

    def reset(self):
        self.due_dqnagent.reset()

    def choose_action(self, state_results):
        return self.due_dqnagent.choose_action(state_results)

    def learn(self, state_results, state_results_next, action_idx):

        reward = ec.reward_function(state_results,state_results_next)
        self.due_dqnagent.put(state_results, action_idx, reward, state_results_next)
        self.due_dqnagent.learn()


########################################################
#print(torch.cuda.current_device())

TARGET_REPLACE_ITER = 50  # target update frequency 100

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size ):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out.weight.data.normal_(0, 0.1)   # initialization

        self.fc2 = nn.Linear(input_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, output_size)
        self.fc2.weight.data.normal_(0,0.1)
        self.out2.weight.data.normal_(0,0.1)

    def forward(self, x):

        x1 = F.relu(self.fc1(x))
        y1 = self.out(x1)

        x2 = F.relu(self.fc2(x))
        y2 = self.out2(x2)
        #val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        #x3 = y1.expand_as(y2) + (y2 - y2.mean(1).expand_as(y2))
        x3 = y1 + (y2 - y2.mean())
        actions_value = x3

        return actions_value
    def reset(self):

        self.out.reset_parameters()
        self.out2.reset_parameters()
 

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

class DuelingDQN(object):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.target_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.buffer = []
        self.learn_step_counter = 0  # for target updating
        self.steps = 0

    def choose_action(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        # input only one sample
        if random.random() < epsi:
            # random
            action = random.randrange(self.action_space_dim)
        else:
            # greedy
            s0 = torch.tensor(s0, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.eval_net(s0)).item()  # return the argmax
        return action

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):

        if (len(self.buffer)) < self.batch_size:
            return

        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        q_target = r1 + self.gamma * torch.max(self.target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        q_eval = self.eval_net(s0).gather(1, a0)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.eval_net.reset()
        self.target_net.reset()
        self.buffer=[]
        self.steps = 0
        self.learn_step_counter = 0