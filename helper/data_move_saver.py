import scipy.io as sio
import numpy as np
from time import strftime
import config.ql_anneal_config as qc
import config.comm_config as cc
import os
from helper.logger_maker import get_logger
#path,filename(time),data
class DataSaver:
    def __init__(self,saver_name):
        self.logger = get_logger("DataSaver_%s" % (saver_name))

        self.saver_name = saver_name
        self.time_saver = strftime("%Y%m%d_%H%M%S")

        self.dir_name = "%s_%s" %(self.saver_name,self.time_saver)
        self.data_dir = os.path.join("results",self.dir_name)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.save_move_destination = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))
        self.save_move_distance = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))
        self.save_packet_loss = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))
        self.save_data_latency = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))
        self.save_move_latency = np.zeros((gc.EPISODE_NUMBER, gc.STEP_NUMBER))
        self.save_wifi_latency = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))
        self.save_channel = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))
        self.save_utility = np.zeros((gc.EPISODE_NUMBER,gc.STEP_NUMBER))

        self.save_dict ={
            "move_destination" : self.save_move_destination,
            "move_distance"    : self.save_move_distance,
            "packet_loss"      : self.save_packet_loss,
            "data_latency"     : self.save_data_latency,
            "move_latency"     : self.save_move_latency,
            "wifi_latency"     : self.save_wifi_latency,
            "channel"          : self.save_channel,
            "utility"          : self.save_utility,
        }

    def save_step(self, episode, step, state_results,state_result_next):
        move_destination, packet_loss, data_latency,move_latency,wifi_latency,channel, move_direction_ = state_result_next
        move_distance =abs(state_results[0]-state_result_next[0])
        self.save_move_destination[episode, step] = move_destination
        self.save_move_distance[episode, step] = move_distance
        self.save_packet_loss[episode, step] = packet_loss
        self.save_data_latency[episode, step] = data_latency
        self.save_move_latency[episode, step] = move_latency
        self.save_wifi_latency[episode, step] = wifi_latency
        self.save_channel[episode, step] = channel
        self.save_utility[episode, step] = qc.reward_function(state_results,state_result_next)
        channel_idx = 0 if channel == 1 else 1

        self.logger.info("EP:%-3d ST:%-4d MD:%02d LO:%-3d DL:%-8.4f CH:%-2d MD:%-2d move_dir: %-2d",
                    episode, step,
                    move_distance, packet_loss, data_latency,channel,move_destination,move_direction_)

    def save_qtable(self, episode, ql_agent):
        qtable_dir = os.path.join(self.data_dir, "Q")
        if not os.path.exists(qtable_dir):
            os.mkdir(qtable_dir)
        qtable_path = os.path.join(qtable_dir, "%d.mat" % (episode))
        ql_agent.ql.save_q(qtable_path)
        self.logger.info("Save Q: %s", qtable_path)

    def save_mat(self):
        save_mat_path = os.path.join(self.data_dir,"result.mat")
        sio.savemat(save_mat_path,self.save_dict)
        self.logger.info(self.dir_name)

