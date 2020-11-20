import scipy.io as sio
import numpy as np
import os
import config.env_config as ec

DATA_DIR = "data"

def load_data():
    ndata_format = os.path.join(DATA_DIR, "n%2d.mat")
    jdata_format = os.path.join(DATA_DIR, "j%2d.mat")
    ndata = [{}] * ec.CHANNEL_NUMBER
    jdata = [{}] * ec.CHANNEL_NUMBER
    for channel_idx in range(ec.CHANNEL_NUMBER):
        channel = ec.CHANNELS[channel_idx]
        ndata_path = ndata_format % (channel)
        jdata_path = jdata_format % (channel)

        ndata_file = sio.loadmat(ndata_path)
        jdata_file = sio.loadmat(jdata_path)
        for entry in ec.DATA_ENTRIES:
            ndata[channel_idx][entry] = ndata_file[entry].squeeze()
            jdata[channel_idx][entry] = jdata_file[entry].squeeze()

    sdata_path = os.path.join(DATA_DIR, "sl.mat")
    sdata_file = sio.loadmat(sdata_path)
    sdata = {}
    for entry in ec.SWITCH_ENTRIES:
        sdata[entry] = sdata_file[entry].squeeze()

    return ndata, jdata, sdata

class EnvData:
    def __init__(self):
        self.ndata, self.jdata, self.sdata = load_data()
        self.data_slot_number = self.ndata[0][ec.DATA_ENTRIES[0]].shape[0]
        self.switch_slot_number = self.sdata["sjn"].shape[0]
        print(self.data_slot_number, self.switch_slot_number)

    def shuffle_data(self):
        data_indices = np.arange(self.data_slot_number)
        np.random.shuffle(data_indices)
        for channel_idx in range(ec.CHANNEL_NUMBER):
            for entry in ec.DATA_ENTRIES:
                self.ndata[channel_idx][entry][:] = self.ndata[channel_idx][entry][data_indices]
                self.jdata[channel_idx][entry][:] = self.jdata[channel_idx][entry][data_indices]

        switch_indices = np.arange(self.switch_slot_number)
        np.random.shuffle(switch_indices)
        for entry in ec.SWITCH_ENTRIES:
            self.sdata[entry][:] = self.sdata[entry][switch_indices]

    def get_data(self, slot, channel_from, channel_to, channel_jammed):
        data_slot = slot % self.data_slot_number
        results = []
        if channel_to == channel_jammed:
            data = self.jdata
        else:
            data = self.ndata
        for entry in ec.DATA_ENTRIES:
            results.append(data[channel_to][entry][data_slot])

        if channel_from == channel_to:
            wifi_latency = 0
        else:
            if channel_to == channel_jammed:
                stype = "snj"
            elif channel_from == channel_jammed:
                stype = "sjn"
            else:
                stype = "snn"
            switch_slot = slot % self.switch_slot_number
            wifi_latency = self.sdata[stype][switch_slot]

        results.append(wifi_latency)
        results.append(channel_to)

        return results # (pktloss, datalat, wifilat, channel)
