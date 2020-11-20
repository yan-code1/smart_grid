import numpy as np


'''
   JAMMER_TYPE : 0 --- no jamming
                 1 --- fixed jamming
                 2 --- switch jamming
'''
JAMMER_TYPE = 2
JAMMER_SWITCH_MEAN_SLOT = 40

CHANNELS = [1, 6] #, 11

MOVE_STEP = 2   #the distance between two adjacent points
# TRANSMIT_CHANNEL_IDX = 0
DATA_LATENCY = [0,0.1,0.21,0.88]
MOVE_DESTINATION =[0,MOVE_STEP,2*MOVE_STEP,3*MOVE_STEP]
MOVE_DIRECTIONS = [-1,0,1]
PACKET_LOSS = 5

# PACKET_LOSS_LEVELS = 5
# DATA_LATENCY_LEVELS = len(DATA_LATENCY)
# WIFI_LATENCY_LEVELS = 2
# MOVE_LATENCY_LEVELS = 2
# MOVE_LEVELS = len(MOVE_DIRECTIONS)

MOVE_DESTINATION_NUM = len(MOVE_DESTINATION)
CHANNEL_NUMBER = len(CHANNELS)
#
# ACTION_NUM = MOVE_LEVELS * CHANNEL_NUMBER
# STATE_NUM = MOVE_LEVELS * \
#             PACKET_LOSS_LEVELS * \
#             DATA_LATENCY_LEVELS * \
#             WIFI_LATENCY_LEVELS * \
#             CHANNEL_NUMBER *\
#             MOVE_LATENCY_LEVELS
# WIN_LENGTH = 5
# def state_result_encode(state_results):
#     move_destination, packet_loss, data_latency , move_latency,wifi_latency,channel,move_direction_= state_results
#     md_idx = int(move_destination/MOVE_STEP)
#     pl_idx = int (packet_loss > PACKET_LOSS)
#     dl_idx = -1
#     for i in range(DATA_LATENCY_LEVELS):
#         if data_latency <= DATA_LATENCY[i]:
#             dl_idx = i
#             break
#
#     ml_idx = int(move_latency>0)
#     wl_idx = int(wifi_latency > 0)
#     channel_idx = 0 if channel == 1 else 1
#     state_idx = md_idx +\
#                 pl_idx*MOVE_LEVELS +\
#                 dl_idx*MOVE_LEVELS*PACKET_LOSS_LEVELS + \
#                 ml_idx*DATA_LATENCY_LEVELS*MOVE_LEVELS*PACKET_LOSS_LEVELS +\
#                 wl_idx*MOVE_LATENCY_LEVELS*DATA_LATENCY_LEVELS*MOVE_LEVELS*PACKET_LOSS_LEVELS + \
#                 channel_idx *WIFI_LATENCY_LEVELS*MOVE_LATENCY_LEVELS* DATA_LATENCY_LEVELS * MOVE_LEVELS * PACKET_LOSS_LEVELS
#     return state_idx
# aa = []
# bb = []
# cc = []
# dd = []
# count = 0
# def reward_function (state_result,state_result_next):
#     global count
#     move_destination, packet_loss, data_latency, move_latency,wifi_latency, channel, move_direction_= state_result_next
#     a=(-abs(move_direction_)*MOVE_STEP)*25
#     b=0*int(packet_loss < PACKET_LOSS)
#     c=-(data_latency)*10
#     d =  -3*wifi_latency
#     count = count+1
#     # aa.append(a)
#     # bb.append(b)
#     # cc.append(c)
#     # dd.append(d)
#
#     # print("a:max:",max(aa),"   min:",min(aa))
#     # print("b:max:", max(bb), "   min:", min(bb))
#     # print("c:max:", max(cc), "   min:", min(cc))
#     # print("d:max:", max(dd), "   min:", min(dd))
#
#     reward =2*int(packet_loss < PACKET_LOSS) - (data_latency)*10  \
#              - 40*wifi_latency - 20*move_latency
#     # if count >= 999:
#     #     print(aa,"\n",bb,"\n",cc,"\n",dd,"\n")
#     return reward
#

#单位移
# ACTION_NUM = MOVE_LEVELS
# STATE_NUM = MOVE_DESTINATION_NUM * \
#             PACKET_LOSS_LEVELS * \
#             DATA_LATENCY_LEVELS
#
#
# def state_result_encode(state_results):
#     move_destination, packet_loss, data_latency,move_direction= state_results
#     ml_idx = int(move_destination/MOVE_STEP)
#     pl_idx = int (packet_loss > PACKET_LOSS)
#     dl_idx = -1
#     for i in range(DATA_LATENCY_LEVELS):
#         if data_latency <= DATA_LATENCY[i]:
#             dl_idx = i
#             break
#
#     state_idx = ml_idx +\
#                 pl_idx*MOVE_LEVELS +\
#                 dl_idx*MOVE_LEVELS*PACKET_LOSS_LEVELS
#     return state_idx
#
# def reward_function (state_result,state_result_next):
#     move_destination, packet_loss, data_latency ,move_direction= state_result_next
#     a=-abs(state_result[0]-state_result_next[0])*0.5
#     b=int(packet_loss<PACKET_LOSS)
#
#
#     reward = (-abs(state_result[0]-state_result_next[0])*0.1) \
#              + 3*int(packet_loss < PACKET_LOSS)-data_latency*10
#
#     return reward