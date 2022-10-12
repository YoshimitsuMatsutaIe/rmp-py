
import numpy as np
from math import sin, cos, pi, sqrt
np.set_printoptions(precision=1)
import copy



import sys
sys.path.append(".")

import robot_sice.sice as sice

cdim = 4
t_dim = 2

#q = np.zeros((cdim, 1))
q = np.random.rand(4, 1)
q_dot = np.zeros((cdim, 1))
l = [1.0] * cdim
#print(l)

def HTM(theta, l):
    return np.array([
        [cos(theta), -sin(theta), l],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])

Lambda = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

def HTM_dot(theta, l):
    return np.array([
        [-sin(theta), -cos(theta), 0],
        [cos(theta), -sin(theta), 0],
        [0, 0, 0]
    ])



# local同時変換行列
T_local = []
for i in range(cdim):
    if i == 0:
        T_local.append(HTM(q[0,0], 0))
    else:
        T_local.append(HTM(q[i,0], l[i-1]))
T_local.append(HTM(0, l[-1]))


# global同時変換行列
T_global = []
for i, T in enumerate(T_local):
    if i == 0:
        T_global.append(T_local[0])
    else:
        T_global.append(T_global[i-1] @ T)
    #print(T_global[-1])



# local同時変換行列の角度微分
dTdq_local = []
for i in range(cdim):
    if i == 0:
        dTdq_local.append(HTM_dot(q[0, 0], 0))
    else:
        dTdq_local.append(HTM_dot(q[i, 0], l[i-1]))
dTdq_local.append(HTM_dot(0, l[-1]))


# dTdq_s = []
# for i in range(cdim):  #joint val
#     tmp_dTdq_s = []
#     for j, T in enumerate(T_local):  #T val
#         if j == 0:
#             if i == 0:
#                 tmp_dTdq_s.append(dTdq_local[0])
#             else:
#                 tmp_dTdq_s.append(T_local[0])
#         else:
#             if j == i:
#                 tmp_dTdq_s.append(tmp_dTdq_s[j-1] @ dTdq_local[j])
#             else:
#                 tmp_dTdq_s.append(tmp_dTdq_s[j-1] @ T)
    
#     dTdq_s.append(tmp_dTdq_s)

dTdq_s = []
for i in range(cdim):
    if i == 0:
        temp_ = [dTdq_local[0]]
    else:
        temp_ = copy.deepcopy(T_global[:i])
        temp_.append(temp_[i-1] @ dTdq_local[i])
    
    for T in T_local[i+1:]:
        temp_.append(temp_[-1] @ T)
    
    dTdq_s.append(temp_)



Jrx = []
Jry = []
Joo = []

for i in range(cdim+1):
    Jrx_ = np.empty((t_dim, cdim))
    Jry_ = np.empty((t_dim, cdim))
    Joo_ = np.empty((t_dim, cdim))
    
    for j in range(cdim):
        T_ = dTdq_s[j][i]
        Jrx_[:, j:j+1] = T_[0:t_dim, 0:1]
        Jry_[:, j:j+1] = T_[0:t_dim, 1:2]
        Joo_[:, j:j+1] = T_[0:t_dim, 2:3]
    
    Jrx.append(Jrx_)
    Jry.append(Jry_)
    Joo.append(Joo_)


### チェック
print("\nchecking rx...")
for i in range(5):
    print("i = ", i)
    print(np.linalg.norm(T_global[i][0:2, 0:1] - sice.rx(q, i)))

print("\nchecking ry...")
for i in range(5):
    print("i = ", i)
    print(np.linalg.norm(T_global[i][0:2, 1:2] - sice.ry(q, i)))

print("\nchecking o...")
for i in range(5):
    print("i = ", i)
    print(np.linalg.norm((T_global[i][0:2, 2:3] - sice.o(q, i, l[0], l[1], l[2], l[3]))))

print("\nchecking jrx...")
for i in range(5):
    print("i = ", i)
    print(np.linalg.norm(Jrx[i] - sice.jrx(q, i)))

print("\nchecking jry...")
for i in range(5):
    print("i = ", i)
    print(np.linalg.norm(Jry[i] - sice.jry(q, i)))

print("\nchecking jo...")
for i in range(5):
    print("i = ", i)
    print(np.linalg.norm(Joo[i] - sice.jo(q, i, l[0], l[1], l[2], l[3])))