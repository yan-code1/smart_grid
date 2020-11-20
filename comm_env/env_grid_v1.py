from comm.data_helper import DataHelper
from comm.picture_display import PictureDisplay
from robot.robot_control import RobotControl
from comm.logger_maker import get_logger

from time import time, sleep
import numpy as np


PICTURE_DELAY = 0.01

class Env:
    def __init__(self, env_config, data_saver):
        self.logger = get_logger("RobotMove")
        self.picture_idx = 0

        self.ec = env_config

        self.dh = DataHelper()
        self.wc = self.dh.create_wifi_control()
        # self.pd = PictureDisplay(data_saver, self.exit_hook)

        self.wc.connect()
        self.pd.start_draw_process()
        self.dh.start_recv_data()

        self.rc = RobotControl()

    def one_picture(self):
        self.dh.start_sending_picture(self.picture_idx)
        picture_idx_list = self.dh.recv_picture_idx()
        sleep(PICTURE_DELAY)

        for idx in picture_idx_list:
            packet_loss, data_latency = self.dh.calc_results(idx)
            self.pd.draw_picture(self.picture_idx, self.dh.get_picture(idx))

        self.picture_idx += 1
        return packet_loss, data_latency

    def reset(self):
        self.picture_idx = 0

        channel = self.ec.CHANNELS[self.ec.TRANSMIT_CHANNEL_IDX]
        self.wc.change_channel(channel)
        self.logger.info("Current channel: %d", channel)

        packet_loss, data_latency = self.one_picture()

        results = {
            "packet_loss": packet_loss,
            "data_latency": data_latency,
        }

        self.logger.debug("Reset: %s", results)
        return results


    def step(self, move_action):
        self.rc.move_axis(move_action)
        
        packet_loss, data_latency = self.one_picture()

        results = {
            "packet_loss": packet_loss,
            "data_latency": data_latency,
        }

        self.logger.debug("Step %d: %s", self.picture_idx, results)

        return results

    def exit_hook(self):
        self.dh.stop_recv_data()
        self.pd.stop_draw_process()
        exit(0)
