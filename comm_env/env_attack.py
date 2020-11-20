import numpy as np
import config.env_robot_config as ec
import config.ql_anneal_config as qc
import  config.global_exp_config as gc
import scipy.io as sio
import random
from comm_env.jammer_v2 import Jammer
import os
DATA_DIR = 'data'

class Env :
    def __init__(self):
        self.jammer = Jammer()
        self.data = [
                     [[sio.loadmat('data/61jp1.mat'), sio.loadmat('data/61jp2.mat'), sio.loadmat('data/61jp3.mat'),sio.loadmat('data/61jp4.mat')],
                      [sio.loadmat('data/61np1.mat'), sio.loadmat('data/61np2.mat'), sio.loadmat('data/61np3.mat'),sio.loadmat('data/61np4.mat')]],

                     [[sio.loadmat('data/16jp1.mat'), sio.loadmat('data/16jp2.mat'), sio.loadmat('data/16jp3.mat'),sio.loadmat('data/16jp4.mat')],
                      [sio.loadmat('data/16np1.mat'), sio.loadmat('data/16np2.mat'), sio.loadmat('data/16np3.mat'),sio.loadmat('data/16np4.mat')]],
                    ]
        self.move_interval = [sio.loadmat('data/mobility_time_interval_data.mat')]

        self.current_move_destination_idx = 1#random.randrange(4)
        self.currunt_channel_idx =random.randrange(ec.CHANNEL_NUMBER)
        self.last_move_dextination_idx = self.current_move_destination_idx
        self.last_channel_idx = self.currunt_channel_idx

        # self.data = [[sio.loadmat('data/Nd1.mat'), sio.loadmat('data/Nd2.mat'), sio.loadmat('data/Nd3.mat'),
        #               sio.loadmat('data/Nd4.mat')],
        #              [sio.loadmat('data/Jd1.mat'), sio.loadmat('data/Jd2.mat'), sio.loadmat('data/Jd3.mat'),
        #               sio.loadmat('data/Jd4.mat')]]


        self.data_slot_num = self.data[0][0][0]['data_latency'].shape[1]
        self.data_episode_num = self.data[0][0][0]['data_latency'].shape[0]
        self.data_repeat_slot = 50
        self.data_slot =  0
        self.data_episode = random.randrange(self.data_episode_num)
        self.channel_type = 0
        self.jammer_type = 0  # 信道1固定有干扰
    def reset(self):
        self.jammer.reset()
        self.jammer_type = 0 if self.jammer.get_jamming_state() else 1
        self.count = 0
        self.last_channel_idx = self.currunt_channel_idx
        self.currunt_channel_idx = 0#np.random.randint(ec.CHANNEL_NUMBER)
        results = self.get_current_result()
        self.data_slot = np.random.randint(50, self.data_slot_num)
        self.data_episode = random.randrange(self.data_episode_num)


        return results

    # [self.currunt_channel_idx][self.jammer_type][self.current_move_destination_idx]
    def get_current_result(self):
        move_destiantion = ec.MOVE_DESTINATION[self.current_move_destination_idx]

        #数据获取
        packet_loss  = self.data[self.currunt_channel_idx][self.jammer_type][self.current_move_destination_idx]['packet_loss'][ self.data_episode][self.data_slot]
        data_latency = self.data[self.currunt_channel_idx][self.jammer_type][self.current_move_destination_idx]['data_latency'][ self.data_episode][self.data_slot]
        wifi_latency = self.data[self.currunt_channel_idx][self.jammer_type][self.current_move_destination_idx]['wifi_latency'][self.data_episode][self.data_slot]
        channel = ec.CHANNELS[self.currunt_channel_idx]
        if self.last_channel_idx == self.currunt_channel_idx:
            wifi_latency=0
        # 移动判断
        if self.last_move_dextination_idx != self.current_move_destination_idx:
            move_latency = self.move_interval[0]['interval'][0][self.data_slot%(len(self.move_interval[0]['interval'][0]))]
        else:
            move_latency =0
        #结果
        # if  self.current_move_destination_idx < 0 or  self.current_move_destination_idx >= ec.MOVE_DESTINATION_NUM:
        #         self.current_move_destination_idx = self.last_move_dextination_idx
        move_direction_ = self.current_move_destination_idx - self.last_move_dextination_idx

        state_result = [move_destiantion , packet_loss, data_latency , move_latency ,wifi_latency  , channel , move_direction_]
        return state_result


    def step(self,action_idx,FLAG=False):
        self.jammer_type = 0 if self.jammer.step() else 1
        self.data_slot += 1
        if self.data_slot >= self.data_repeat_slot:
            random_slot = np.random.randint(self.data_slot_num - self.data_repeat_slot)
            self.data_slot = self.data_repeat_slot + random_slot
        self.data_episode = random.randrange(self.data_episode_num)
        # self.data_slot = np.random.randint(50, self.data_slot_num)
        self.last_move_dextination_idx = self.current_move_destination_idx
        self.last_channel_idx = self.currunt_channel_idx

        self.currunt_channel_idx = 0 if action_idx < qc.ACTION_NUM/ec.CHANNEL_NUMBER else 1
        move_direction_idx = action_idx if action_idx<qc.ACTION_NUM/ec.CHANNEL_NUMBER else action_idx-qc.ACTION_NUM/ec.CHANNEL_NUMBER


        if FLAG: #进行了决策
            real_action_idx = action_idx
            self.move_direction = ec.MOVE_DIRECTIONS[int(move_direction_idx)]
            self.current_move_destination_idx += self.move_direction
            if  self.current_move_destination_idx < 0 or  self.current_move_destination_idx >= ec.MOVE_DESTINATION_NUM:
                    self.current_move_destination_idx = self.last_move_dextination_idx
        else:
            real_action_idx = self.last_channel_idx * qc.MOVE_LEVELS + move_direction_idx
            self.currunt_channel_idx = self.last_channel_idx
            self.current_move_destination_idx = self.last_move_dextination_idx

        if self.last_channel_idx != self.currunt_channel_idx:
            self.data_slot = 0

        results = self.get_current_result()

        return results,int(real_action_idx)
