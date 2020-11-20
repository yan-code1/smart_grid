import pylab
import scipy.io as sio
import config.global_exp_config as gc
from algorithms.agent.ql_Agent_anneal import QL_Agent
from helper.logger_maker import get_logger
from tqdm import tqdm
from time import strftime
import config.ql_anneal_config as qc
# from comm_env.env_move import Env
from helper.data_move_saver import DataSaver
import numpy as np
import random
import os

# H=[[0,0.8,0.5],
#    [0.3,0,0.6],
#    [0.2,0.2,0]]
H=[[21.38,-16.90 ,0.00 ,0.00 ,-4.48 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[-16.90 ,33.37 ,-5.05 ,-5.67 ,-5.75 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,-5.05 ,10.90 ,-5.85 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,-5.67 ,-5.85 ,42.01 ,-23.75 ,0.00 ,-4.89 ,0.00 ,-1.86 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[-4.48 ,-5.75 ,0.00 ,-23.75 ,38.24 ,-4.26 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,-4.26 ,20.87 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.03 ,-3.91 ,-7.68 ,0.00],
[0.00 ,0.00 ,0.00 ,-4.89 ,0.00 ,0.00 ,19.66 ,-5.68 ,-9.09 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.68 ,5.68 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,-1.86 ,0.00 ,0.00 ,-9.09 ,0.00 ,26.48 ,-11.83 ,0.00 ,0.00 ,0.00 ,-3.70],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-11.83 ,17.04 ,-5.21 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.03 ,0.00 ,0.00 ,0.00 ,-5.21 ,10.23 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-3.91 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,8.91 ,-5.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-7.68 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.00 ,15.55 ,-2.87],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-3.70 ,0.00 ,0.00 ,0.00 ,-2.87 ,6.57],
[16.90 ,-16.90 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[4.48 ,0.00 ,0.00 ,0.00 ,-4.48 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,5.05 ,-5.05 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,5.67 ,0.00 ,-5.67 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,5.75 ,0.00 ,0.00 ,-5.75 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,5.85 ,-5.85 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,23.75 ,-23.75 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,4.89 ,0.00 ,0.00 ,-4.89 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,1.86 ,0.00 ,0.00 ,0.00 ,0.00 ,-1.86 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,4.26 ,-4.26 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,5.03 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.03 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,3.91 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-3.91 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,7.68 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-7.68 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,5.68 ,-5.68 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,9.09 ,0.00 ,-9.09 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,11.83 ,-11.83 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,3.70 ,0.00 ,0.00 ,0.00 ,0.00 ,-3.70],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,5.21 ,-5.21 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,5.00 ,-5.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,2.87 ,-2.87],
[-16.90 ,16.90 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[-4.48 ,0.00 ,0.00 ,0.00 ,4.48 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,-5.05 ,5.05 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,-5.67 ,0.00 ,5.67 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,-5.75 ,0.00 ,0.00 ,5.75 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,-5.85 ,5.85 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,-23.75 ,23.75 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,-4.89 ,0.00 ,0.00 ,4.89 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,-1.86 ,0.00 ,0.00 ,0.00 ,0.00 ,1.86 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,-4.26 ,4.26 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.03 ,0.00 ,0.00 ,0.00 ,0.00 ,5.03 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-3.91 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,3.91 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-7.68 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,7.68 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.68 ,5.68 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-9.09 ,0.00 ,9.09 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-11.83 ,11.83 ,0.00 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-3.70 ,0.00 ,0.00 ,0.00 ,0.00 ,3.70],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.21 ,5.21 ,0.00 ,0.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-5.00 ,5.00 ,0.00],
[0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,-2.87 ,2.87]]

# c= np.random.normal(loc=0,scale=1,size=(3,3))

c= np.random.normal(loc=0,scale=1,size=(3,14))
#####env parameters##########
meters_num = 54
meters_num_control = 4
attack_data_set = [0,1]
power_flow_set=[-22.6712, 7.5712, 3.2987, 16.6034, 17.4267, 13.2987, 2.4373,
                9.7629, 5.6019, 19.8352, -1.6965, 9.0144, 8.5173, 0.0000, 9.7629,
                7.6965, 5.5683,5.1965,-5.9856,-4.5683]

# power_flow_set = [[5,7,10],[3,5,6],[1,3,5]]#meter measurements (1,2,3)
detection_probability = []


ATTACK_DATA_NUM = len (attack_data_set)
power_flow = np.zeros(meters_num)
attack_interval_epi = 5
detection_proability_interval = 10
threshlod = 2
#ql quantification
PROBABILITY_Q = [0,0.5,0.85]
PF_MIN,PF_MAX = qc.quantify_PF(power_flow_set)
num_length = int(PF_MAX-PF_MIN)
POWER_FLOW_Q = np.linspace(PF_MIN,PF_MAX,num_length)

#initial RL parameters
c0 =1
EPISODE_NUMBER =3
STEP_NUMBER = 3000
iter_para_times = 100 #迭代次数  DDQN



ACTION_NUM = len(attack_data_set)**meters_num
STATE_NUM = len(PROBABILITY_Q)*len(POWER_FLOW_Q)
#save results
save_suc_probablity = np.zeros((EPISODE_NUMBER,STEP_NUMBER))
save_data_injection = np.zeros((EPISODE_NUMBER,STEP_NUMBER))
save_power_flow = np.zeros((EPISODE_NUMBER,STEP_NUMBER))
save_utility = np.zeros((EPISODE_NUMBER,STEP_NUMBER))

save_dict ={
            "suc_probablity" : save_suc_probablity,
            "data_injection" : save_data_injection,
            "power_flow"     : save_power_flow,
            "utility"        : save_utility,
             }

def attack_detection(data_injection,threshlod,H):
    state_deviation = data_injection-np.dot(H,c)
    a=np.linalg.norm(state_deviation, ord=2)
    if np.linalg.norm(state_deviation, ord=2) > threshlod:
        return True
    else:
        return False

# def get_detection_threshold():
#     step = 0.1
#     while(True):
#         data_injection[0] += step
#         attack_det_result = attack_detection()
#         if attack_det_result:
#             datection_threshold = threshlod_calculate(data_injection)
#             return datection_threshold
def get_detection_probablity(detection_result_T):  #已测试
    sum =0
    d_length = len(detection_result_T)
    for i in range(d_length):
        sum += int(detection_result_T[d_length-1-i])
    return float(sum)/d_length

# def get_price(power_flow,price):###简化版
#     quantification = [0,7,9,11]
#     for i in range(len(quantification)):
#         if np.linalg.norm(power_flow, ord=2) < quantification[i]:
#             return price[i]


def reward_function(state,data_injection):
    power_flow, suc_probablity = state
    attack_cost = np.linalg.norm(data_injection, ord=2)
    reward = c0 * suc_probablity -attack_cost
    return reward

# def action_idx_encode(data_injection,attack_data_set): #已测试
#     action_idx = 0
#     action_encode = np.zeros(len(data_injection))
#     for i in len(data_injection):
#         for j in range (len(attack_data_set)):
#             if data_injection[i]<=attack_data_set[j]:
#                 action_encode[i] = j
#                 break
#     for i in len(action_encode):
#         action_idx += (i+1)*action_encode[i]
#     return action_idx

def state_encode (state):
    power_flow, suc_probablity = state
    pf_idx = -1
    for i in range(len(POWER_FLOW_Q)):
        if np.linalg.norm(power_flow,ord=2) <= POWER_FLOW_Q[i]:
            pf_idx = i
    sp_idx = -1
    for i in range(len(PROBABILITY_Q)):
        if suc_probablity <= PROBABILITY_Q[i]:
            sp_idx = i
    state_idx = sp_idx + pf_idx * len(PROBABILITY_Q)
    return state_idx
def get_data_injection(action_idx):#已测试
    max_idx = (ATTACK_DATA_NUM-1)*(ATTACK_DATA_NUM**meters_num-1)
    data_injection = np.zeros(meters_num)
    # print(max_idx)
    if action_idx > max_idx or action_idx < 0:
        print("action_idx error ...")
        return
    for i in range(meters_num):
        data_injection[i] = attack_data_set[int(action_idx % (ATTACK_DATA_NUM))]
        action_idx /= ATTACK_DATA_NUM
    # print(data_injection)
    return data_injection


def save_mat():
    time_saver = strftime("%Y%m%d_%H%M%S")
    path = "./" + 'smart_grid' + "%s" % (time_saver)
    save_mat_path = os.path.join(path, "result.mat")
    if not os.path.exists(path):
        os.mkdir(path)
    sio.savemat(save_mat_path, save_dict)

def results_plot(save_dict):
    i=0
    for key in save_dict:
        pylab.figure(i)
        pylab.plot(range(STEP_NUMBER),save_dict[key].mean(axis=0), label=key)
        pylab.legend(loc='best')
        pylab.title('')
        pylab.xlabel('Time slot')
        pylab.ylabel(key)
        pylab.grid(True, linestyle='-.')
        pylab.show()
        i = i+1


CODE_VERSION = "ql"
def main():
    agent = QL_Agent(STATE_NUM,ACTION_NUM)
    logger=get_logger("DataSaver_%s" % (CODE_VERSION))
    ###initial env
    power_flow = np.zeros(meters_num)
    for i in range(meters_num):
        power_flow[i] = power_flow_set[i][0] #初始化为最低的电表测量电流
    power_flow_temp = power_flow
    data_injection = np.zeros(meters_num)
    last_injection = np.zeros(meters_num)
    for episode in range(EPISODE_NUMBER):
        detection_result_T=[]
        suc_probablity = 0 #起初无攻击，故攻击成功率为0
        agent.reset()
        for step in tqdm(range(STEP_NUMBER)):
            state = [power_flow, suc_probablity]
            state_idx = state_encode(state)
            action_idx = random.randrange(ACTION_NUM)
            # action_idx = agent.choose_action(state_idx)
            data_injection = get_data_injection(action_idx)
            #若选择不注入数据，则使用上次注入数据到电表
            if np.linalg.norm(data_injection, ord=2) == 0:
                data_injection = last_injection
                print(data_injection)
            last_injection = data_injection

            # 增量注入
            # power_flow_next = power_flow+data_injection
            #收敛注入
            power_flow_next = power_flow_temp+data_injection
            #####原版概率
            detection_result_T.append(attack_detection(data_injection, threshlod,H))
            if step < attack_interval_epi:
                suc_probablity_next = 1-sum(detection_result_T)/(float)(step+1)
            else:
                suc_probablity_next = 1-get_detection_probablity(detection_result_T)
                detection_result_T.remove(detection_result_T[0])
            #####简化版 仅含0和1，不考虑平均
            # suc_probablity_next = 1 - int(attack_detection(data_injection, threshlod,H))
            state_next = [power_flow_next,suc_probablity_next]
            
            reward = reward_function(state_next,data_injection)
            # ql state
            state_next_idx = state_encode(state_next)

            agent.learn(state_idx,state_next_idx,action_idx,reward)
            ###update###
            power_flow = power_flow_next
            suc_probablity = suc_probablity_next

           ###save results
            save_suc_probablity[episode][step] = suc_probablity
            save_data_injection[episode][step] = np.linalg.norm(data_injection,ord=2)
            save_power_flow[episode][step] =  np.linalg.norm(power_flow,ord=2)
            save_utility[episode][step] = reward
            logger.info("EP:%-3d ST:%-4d SP:%-2f DI:%-2f PF:%-2f RE:%-2f ",
                        episode, step,
                        suc_probablity, np.linalg.norm(data_injection,ord=2), np.linalg.norm(power_flow,ord=2), reward)
    save_mat()
    results_plot({
            "suc_probablity" : save_suc_probablity,
            "data_injection" : save_data_injection,
            "power_flow"     : save_power_flow,
            "utility"        : save_utility,
             })

if __name__ == '__main__':
    main()
