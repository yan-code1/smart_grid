import numpy as np

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

