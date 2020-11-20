import math
import numpy as np
data_injection = [0,0,0,0]
meters_num = 4
attack_data_set = [0,1,2]
ATTACK_DATA_NUM = len(attack_data_set)
power_flow_set = [[5,7,10],[3,5,6],[1,3,5]]
def quantify_PF(power_flow_set):
    norm_2 = []
    a=[0,0,0]
    for i in range(3):
       for j in range(3):
           for k in range(3):
               a[1] = power_flow_set[1][j]
               a[0] = power_flow_set[0][i]
               a[2] = power_flow_set[2][k]
               norm_2.append(np.linalg.norm(a, ord=2))
               # print(np.linalg.norm(a, ord=2))
               # print('\n')
    return min(norm_2),max(norm_2)
    # for i in range(meters_num):
    #     data_injection[i] = attack_data_set[int(action_idx%(ATTACK_DATA_NUM))]
    #     action_idx /= ATTACK_DATA_NUM
    # return data_injection
        #
        # for j in range(ATTACK_DATA_NUM):
        # if action_idx >= ATTACK_DATA_NUM**(meters_num-1-i):
        #     data_injection[meters_num - 1 - i] = attack_data_set[ATTACK_DATA_NUM-j]
        #     action_idx -= ATTACK_DATA_NUM ** (meters_num - 1 - i)
        # else:
        #     data_injection[meters_num - 1 - i] =attack_data_set[0]

    # return data_injection
# def get_data_injection(action_idx):
#     max_idx = (ATTACK_DATA_NUM-1)*(ATTACK_DATA_NUM**meters_num-1)
#     print(max_idx)
#     if action_idx > max_idx or action_idx < 0:
#         print("input error ...")
#         return
#     for i in range(meters_num):
#         data_injection[i] = attack_data_set[int(action_idx % (ATTACK_DATA_NUM))]
#         action_idx /= ATTACK_DATA_NUM
    return data_injection

def get_detection_probablity(detection_result_T):
    sum =0
    d_length = len(detection_result_T)
    for i in range(d_length):
        sum += int(detection_result_T[i])
    return float(sum)/d_length
def main():
    detection_result_T= [True,False,False,False,True]
#     # e= [[1,0,0],[0,1,0],[0,0,1]]
#     # suc_probablity = 1 - sum(detection_result_T)
#     # a=np.dot(np.array(power_flow_set),np.array(e))
#     # print(a)
#     # print(a.tolist())
#     print(quantify_PF(power_flow_set))
    # get_data_injection()
    # for i in range(meters_num):
    #      data_injection = [1,1,0,0]
    #      print(np.linalg.norm(data_injection, ord=2))
    print(np.linalg.norm([0,0,0,0], ord=2))
if __name__ == '__main__':
    main()

