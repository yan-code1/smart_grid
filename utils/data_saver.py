import scipy.io as sio
import numpy as np
from time import strftime
import config.global_exp_config as gc
import os
import cv2
from comm.logger_maker import get_logger

class DataSaver:
    def __init__(self, saver_name):
        self.logger = get_logger("DataSaver_%s" % (saver_name))

        results_mat_dir = os.path.join("results", "mat")
        if not os.path.exists(results_mat_dir):
            os.mkdir(results_mat_dir)

        results_img_dir = os.path.join("results", "img")
        if not os.path.exists(results_img_dir):
            os.mkdir(results_img_dir)

        self.saver_name = saver_name
        self.time_str = strftime("%Y%m%d_%H%M%S")
        self.dir_name = "%s_%s" % (self.saver_name, self.time_str)

        self.mat_dir = os.path.join(results_mat_dir, self.dir_name)
        if not os.path.exists(self.mat_dir):
            os.mkdir(self.mat_dir)

        self.pic_dir = os.path.join(results_img_dir, self.dir_name)
        if not os.path.exists(self.pic_dir):
            os.mkdir(self.pic_dir)

        self.save_array = {}

    def save_step(self, episode, step, state_results, **other_results):
        for k in state_results.keys():
            if k not in self.save_array:
                data_size = [gc.EPISODE_NUMBER, gc.STEP_NUMBER]
                if type(state_results[k]) is np.ndarray:
                    data_size.extend(state_results[k].shape)
                self.save_array[k] = np.zeros(data_size)
            self.save_array[k][episode, step] = state_results[k]

        for k in other_results.keys():
            if k not in self.save_array:
                data_size = [gc.EPISODE_NUMBER, gc.STEP_NUMBER]
                if type(other_results[k]) is np.ndarray:
                    data_size.extend(other_results[k].shape)
                self.save_array[k] = np.zeros(data_size)
            self.save_array[k][episode, step] = other_results[k]

        self.logger.info("EP:%-3d ST:%-4d %s %s",
                    episode, step, state_results, other_results)

    def save_agent(self, episode, ql_agent):
        if "ql" not in ql_agent.__dict__:
            return
        qtable_dir = os.path.join(self.data_dir, "Q")
        if not os.path.exists(qtable_dir):
            os.mkdir(qtable_dir)
        qtable_path = os.path.join(qtable_dir, "%d.mat" % (episode))
        ql_agent.ql.save_q(qtable_path)
        self.logger.info("Save Q: %s", qtable_path)

    def save_mat(self):
        save_mat_path = os.path.join(self.mat_dir, "results.mat")
        sio.savemat(save_mat_path, self.save_array)
        self.logger.info(self.dir_name)

    # def save_picture(self, picture_idx, picture_data):
    #     picture_path = os.path.join(self.pic_dir, "%04d.jpg" % (picture_idx))
    #     cv2.imwrite(picture_path, picture_data)
