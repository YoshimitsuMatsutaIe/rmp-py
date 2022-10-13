
import numpy as np
from math import sin, cos, pi, sqrt
np.set_printoptions(precision=1)
import copy

from numba import njit

import sys
sys.path.append(".")

import mappings
#import robot_sice.sice as sice

c_dim = 4
t_dim = 2

@njit("f8[:,:](f8, f8)")
def HTM(theta, l):
    return np.array([
        [cos(theta), -sin(theta), l],
        [sin(theta), cos(theta), 0.],
        [0., 0., 1.]
    ])

@njit("f8[:,:](f8, f8)")
def HTM_dot(theta, l):
    return np.array([
        [-sin(theta), -cos(theta), 0.],
        [cos(theta), -sin(theta), 0.],
        [0., 0., 0.]
    ])

@njit("Tuple((f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:]))(i8, f8[:,:], f8[:,:], f8[:])")
def func(n, q, dq, l):
    # local同時変換行列
    T_local = np.empty((t_dim+1, (c_dim+1)*3))
    for i in range(c_dim):
        if i == 0:
            T_local[:, 0:3] = HTM(q[0, 0], 0)
        else:
            T_local[:, 3*i:3*(i+1)] = HTM(q[i, 0], l[i-1])
    T_local[:, c_dim*3:] = HTM(0, l[c_dim-1])


    # global同時変換行列
    T_global = np.empty((t_dim+1, (c_dim+1)*3))
    for i in range(c_dim+1):
        if i == 0:
            T_global[:, 0:3] = T_local[:, 0:3]
        else:
            T_global[:, 3*i:3*(i+1)] = T_global[:, 3*(i-1):3*i] @ T_local[:, 3*i:3*(i+1)]


    # local同時変換行列の角度微分
    dTdq_local = np.empty((t_dim+1, (c_dim+1)*3))
    for i in range(c_dim):
        if i == 0:
            dTdq_local[:, 0:3] = HTM_dot(q[0, 0], 0)
        else:
            dTdq_local[:, 3*i:3*(i+1)] = HTM_dot(q[i, 0], l[i-1])
    dTdq_local[:, 3*c_dim:3*(c_dim+1)] = HTM_dot(0, l[c_dim-1])


    dTdq_s = np.empty((c_dim*(t_dim+1), (c_dim+1)*3))
    for i in range(c_dim):  #joint val
        tmp = np.empty((t_dim+1, (c_dim+1)*3))
        for j in range(c_dim+1):  #T val
            if j == 0:
                if i == 0:
                    tmp[:, 0:3] = dTdq_local[:, 0:3]
                else:
                    tmp[:, 0:3] = T_local[:, 0:3]
            else:
                if j == i:
                    tmp[:, 3*j:3*(j+1)] = tmp[:, 3*(j-1):3*j] @ dTdq_local[:, 3*j:3*(j+1)]
                else:
                    tmp[:, 3*j:3*(j+1)] = tmp[:, 3*(j-1):3*j] @ T_local[:, 3*j:3*(j+1)]
        
        dTdq_s[3*i:3*(i+1), :] = tmp


    Jrx = np.empty((t_dim, c_dim*(c_dim+1)))
    Jry = np.empty((t_dim, c_dim*(c_dim+1)))
    Joo = np.empty((t_dim, c_dim*(c_dim+1)))

    for i in range(c_dim+1):
        Jrx_ = np.zeros((t_dim, c_dim))
        Jry_ = np.zeros((t_dim, c_dim))
        Joo_ = np.zeros((t_dim, c_dim))
        
        for j in range(i):
            T_ = dTdq_s[3*j:3*(j+1), 3*i:3*(i+1)]
            Jrx_[:, j:j+1] = T_[0:t_dim, 0:1]
            Jry_[:, j:j+1] = T_[0:t_dim, 1:2]
            Joo_[:, j:j+1] = T_[0:t_dim, 2:3]
        
        Jrx[:, c_dim*i:c_dim*(i+1)] = Jrx_
        Jry[:, c_dim*i:c_dim*(i+1)] = Jry_
        Joo[:, c_dim*i:c_dim*(i+1)] = Joo_
    
    rx = T_global[0:2, n+0:n+1]
    ry = T_global[0:2, n+1:n+2]
    oo = T_global[0:2, n+2:n+3]
    jrx = Jrx[:, c_dim*n:c_dim*(n+1)]
    jry = Jry[:, c_dim*n:c_dim*(n+1)]
    joo = Joo[:, c_dim*n:c_dim*(n+1)]
    
    return rx, ry, oo, jrx, jry, joo



def q_neutral():
    return np.array([[0 for _ in range(c_dim)]]).T * np.pi/180  # ニュートラルの姿勢

def q_min():
    return np.array([[-90 for _ in range(c_dim)]]).T * np.pi/180

def q_max():
    return -q_min()


class CPoint(mappings.Identity):

    t_dim = t_dim
    c_dim = c_dim

    # 追加
    RS_ALL = (
        ((0, 0),),
        ((0, 0),),
        ((0, 0),),
        ((0, 0),),
        ((0, 0),),
        # ((0, 0),),
        # ((0, 0),),
        # ((0, 0),),
        # ((0, 0),),
        # ((0, 0),),
    )
    ee_id = (c_dim, 0)

    def __init__(self, frame_num, position_num, total_length=4.0):
        self.frame_num = frame_num
        
        self.ls = np.linspace(0, total_length, c_dim+1)[1:]
        
        self.r = self.RS_ALL[frame_num][position_num]
        

    # def __init__(self, frame_num, position_num, c_dim=4, ls=None):
    #     self.frame_num = frame_num
    #     self.c_dim = c_dim
    #     if ls is None:
    #         self.ls = [1 for _ in range(c_dim)]
    #     else:
    #         assert len(ls) != c_dim
    #         self.ls = ls
        
    #     self.r = self.RS_ALL[frame_num][position_num]
    #     self.ee_id = (c_dim, 0)
    
    def calc_all(self, q, dq):
        rx, ry, oo, jrx, jry, joo = func(self.frame_num, q, dq, self.ls)
        
        x = rx * self.r[0] + ry * self.r[1] + oo
        J = jrx*self.r[0] + jry*self.r[1] + joo
        x_dot = J @ dq
        J_dot = np.zeros(J.shape)
        
        return x, x_dot, J, J_dot


    def calc_o(self, n, q):
        _, _, out, _, _, _ = func(n, q, np.zeros(q.shape), self.ls)


def JOINT_PHI():
    h = CPoint(0, 0)
    #return (lambda q: h.calc_o(i, q) for i in range(c_dim+1))

    return (
        lambda q: h.calc_o(0, q),
        lambda q: h.calc_o(1, q),
        lambda q: h.calc_o(2, q),
        lambda q: h.calc_o(3, q),
        lambda q: h.calc_o(4, q),
        # lambda q: h.calc_o(5, q),
        # lambda q: h.calc_o(6, q),
        # lambda q: h.calc_o(7, q),
        # lambda q: h.calc_o(8, q),
        # lambda q: h.calc_o(9, q),
    )



if __name__ == "__main__":
    pass
    # ### チェック
    # # print("\nchecking rx...")
    # # for i in range(5):
    # #     print("i = ", i)
    # #     print(np.linalg.norm(T_global[i][0:2, 0:1] - sice.rx(q, i)))

    # # print("\nchecking ry...")
    # # for i in range(5):
    # #     print("i = ", i)
    # #     print(np.linalg.norm(T_global[i][0:2, 1:2] - sice.ry(q, i)))

    # # print("\nchecking o...")
    # # for i in range(5):
    # #     print("i = ", i)
    # #     print(np.linalg.norm((T_global[i][0:2, 2:3] - sice.o(q, i, l[0], l[1], l[2], l[3]))))

    # print("\nchecking jrx...")
    # for i in range(5):
    #     print("i = ", i)
    #     print("tmp = \n", Jrx[i])
    #     print("true = \n", sice.jrx(q, i))
    #     print(np.linalg.norm(Jrx[i] - sice.jrx(q, i)))

    # print("\nchecking jry...")
    # for i in range(5):
    #     print("i = ", i)
    #     print("tmp = \n", Jry[i])
    #     print("true = \n", sice.jry(q, i))
    #     print(np.linalg.norm(Jry[i] - sice.jry(q, i)))

    # print("\nchecking jo...")
    # for i in range(5):
    #     print("i = ", i)
    #     print("tmp = \n", Joo[i])
    #     print("true = \n", sice.jo(q, i, l[0], l[1], l[2], l[3]))
    #     print(np.linalg.norm(Joo[i] - sice.jo(q, i, l[0], l[1], l[2], l[3])))