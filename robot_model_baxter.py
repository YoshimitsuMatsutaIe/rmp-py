import numpy as np


import mappings

import sys
sys.path.append('.')

from baxter.htm import *
from baxter.Jos import *
from baxter.JRxs import *
from baxter.JRys import *
from baxter.JRzs import *
from baxter.Jo_dots import *
from baxter.JRx_dots import *
from baxter.JRy_dots import *
from baxter.JRz_dots import *


def htm_0(q):
    return np.block([
        [rx_0(q), ry_0(q), rz_0(q), o_0(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_1(q):
    return np.block([
        [rx_1(q), ry_1(q), rz_1(q), o_1(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_2(q):
    return np.block([
        [rx_2(q), ry_2(q), rz_2(q), o_2(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_3(q):
    return np.block([
        [rx_3(q), ry_3(q), rz_3(q), o_3(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_4(q):
    return np.block([
        [rx_4(q), ry_4(q), rz_4(q), o_4(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_5(q):
    return np.block([
        [rx_5(q), ry_5(q), rz_5(q), o_5(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_6(q):
    return np.block([
        [rx_6(q), ry_6(q), rz_6(q), o_6(q)],
        [np.array([[0, 0, 0, 1]])]
    ])

def htm_ee(q):
    return np.block([
        [rx_ee(q), ry_ee(q), rz_ee(q), o_ee(q)],
        [np.array([[0, 0, 0, 1]])]
    ])


class Common:
    L = 278e-3
    h = 64e-3
    H = 1104e-3
    L0 = 270.35e-3
    L1 = 69e-3
    L2 = 364.35e-3
    L3 = 69e-3
    L4 = 374.29e-3
    L5 = 10e-3
    L6 = 368.3e-3

    q_neutral = np.array([[0, -31, 0, 43, 0, 72, 0]]).T * np.pi/180  # ニュートラルの姿勢
    q_min = np.array([[-141, -123, -173, -3, -175, -90, -175]]).T * np.pi/180
    q_max = np.array([[51, 60, 173, 150, 175, 120, 175]]).T * np.pi/180

    # 制御点のローカル座標
    R = 0.05

    r_bars_in_0 = (
        np.array([[0, L1/2, -L0/2, 1]]).T,
        np.array([[0, -L1/2, -L0/2, 1]]).T,
        np.array([[L1/2, 0, -L0/2, 1]]).T,
        np.array([[-L1/2, 0, -L0/2, 1]]).T,
    )  # 1座標系からみた制御点位置

    r_bars_in_1 = (
        np.array([[0, 0, L3/2, 1]]).T,
        np.array([[0, 0, -L3/2, 1]]).T,
    )

    r_bars_in_2 = (
        np.array([[0, L3/2, -L2*2/3, 1]]).T,
        np.array([[0, -L3/2, -L2*2/3, 1]]).T,
        np.array([[L3/2, 0, -L2*2/3, 1]]).T,
        np.array([[-L3/2, 0, -L2*2/3, 1]]).T,
        np.array([[0, L3/2, -L2*1/3, 1]]).T,
        np.array([[0, -L3/2, -L2*1/3, 1]]).T,
        np.array([[L3/2, 0, -L2*1/3, 1]]).T,
        np.array([[-L3/2, 0, -L2*1/3, 1]]).T,
    )

    r_bars_in_3 = (
        np.array([[0, 0, L3/2, 1]]).T,
        np.array([[0, 0, -L3/2, 1]]).T,
    )

    r_bars_in_4 = (
        np.array([[0, R/2, -L4/3, 1]]).T,
        np.array([[0, -R/2, -L4/3, 1]]).T,
        np.array([[R/2, 0, -L4/3, 1]]).T,
        np.array([[-R/2, 0, -L4/3, 1]]).T,
        np.array([[0, R/2, -L4/3*2, 1]]).T,
        np.array([[0, -R/2, -L4/3*2, 1]]).T,
        np.array([[R/2, 0, -L4/3*2, 1]]).T,
        np.array([[-R/2, 0, -L4/3*2, 1]]).T,
    )

    r_bars_in_5 = (
        np.array([[0, 0, L5/2, 1]]).T,
        np.array([[0, 0, -L5/2, 1]]).T,
    )

    r_bars_in_6 = (
        np.array([[0, R/2, L6/2, 1]]).T,
        np.array([[0, -R/2, L6/2, 1]]).T,
        np.array([[R/2, 0, L6/2, 1]]).T,
        np.array([[-R/2, 0, L6/2, 1]]).T,
    )

    r_bars_in_GL = (
        np.array([[0, 0, 0, 1]]).T,
    )

    # 追加
    R_BARS_ALL = (
        r_bars_in_0, r_bars_in_1, r_bars_in_2, r_bars_in_3, r_bars_in_4, r_bars_in_5, r_bars_in_6, r_bars_in_GL,
    )

    r_bar_zero = np.array([[0, 0, 0, 1]]).T

    HTM = (htm_0, htm_1, htm_2, htm_3, htm_4, htm_5, htm_6, htm_ee)
    JO = (jo_0, jo_1, jo_2, jo_3, jo_4, jo_5, jo_6, jo_ee)
    JRX = (jrx_0, jrx_1, jrx_2, jrx_3, jrx_4, jrx_5, jrx_6, jrx_ee)
    JRY = (jry_0, jry_1, jry_2, jry_3, jry_4, jry_5, jry_6, jry_ee)
    JRZ = (jrz_0, jrz_1, jrz_2, jrz_3, jrz_4, jrz_5, jrz_6, jrz_ee)
    JO_DOT = (jo_0_dot, jo_1_dot, jo_2_dot, jo_3_dot, jo_4_dot, jo_5_dot, jo_6_dot, jo_ee_dot)
    JRX_DOT = (jrx_0_dot, jrx_1_dot, jrx_2_dot, jrx_3_dot, jrx_4_dot, jrx_5_dot, jrx_6_dot, jrx_ee_dot)
    JRY_DOT = (jry_0_dot, jry_1_dot, jry_2_dot, jry_3_dot, jry_4_dot, jry_5_dot, jry_6_dot, jry_ee_dot)
    JRZ_DOT = (jrz_0_dot, jrz_1_dot, jrz_2_dot, jrz_3_dot, jrz_4_dot, jrz_5_dot, jrz_6_dot, jrz_ee_dot)


class CPoint(mappings.Id):
    # C = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    # ])
    
    def __init__(self, flame_num, position_num):
        self.htm = Common.HTM[flame_num]
        self.jo = Common.JO[flame_num]
        self.jrx = Common.JRX[flame_num]
        self.jry = Common.JRY[flame_num]
        self.jrz = Common.JRZ[flame_num]
        self.jo_dot = Common.JO_DOT[flame_num]
        self.jrx_dot = Common.JRX_DOT[flame_num]
        self.jry_dot = Common.JRY_DOT[flame_num]
        self.jrz_dot = Common.JRZ_DOT[flame_num]
        self.r_bar = Common.R_BARS_ALL[flame_num][position_num]
    
    def phi(self, q):
        return (self.htm(q) @ self.r_bar)[:3, :]
    
    def J(self, q):
        return (self.jrx(q)*self.r_bar[0,0] + self.jry(q)*self.r_bar[1,0] + self.jrz(q)*self.r_bar[2,0] + self.jo(q))

    def J_dot(self, q, dq):
        return (self.jrx_dot(q, dq)*self.r_bar[0,0] + self.jry_dot(q, dq)*self.r_bar[1,0] + self.jrz_dot(q, dq)*self.r_bar[2,0] + self.jo_dot(q, dq))






if __name__ == "__main__":
    q = Common.q_neutral
    dq = np.zeros_like(q)+0.1
    
    
    
    c = CPoint(6, 2)
    
    print(c.phi(q))
    print(c.J(q))
    print(c.J_dot(q, dq))
    print(c.velocity(c.J_dot(q, dq), dq))