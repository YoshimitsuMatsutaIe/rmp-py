import numpy as np
import mappings

import sys
sys.path.append('.')

from franka_emika.htm import *
from franka_emika.Jos import *
from franka_emika.JRxs import *
from franka_emika.JRys import *
from franka_emika.JRzs import *
from franka_emika.Jo_dots import *
from franka_emika.JRx_dots import *
from franka_emika.JRy_dots import *
from franka_emika.JRz_dots import *


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




class CPoint(mappings.Identity):
    q_neutral = np.array([[0, 0, 0, 0, 0, 0, 0]]).T * np.pi/180  # ニュートラルの姿勢
    q_min = np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]]).T
    q_max = np.array([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]]).T

    # 制御点のローカル座標


    d1 = 0.333
    d3 = 0.316
    d5 = 0.384
    df = 0.107
    a4 = 0.0825
    a5 = -0.0825
    a7 = 0.088


    R = a7

    r_bars_in_0 = (
        np.array([[R/2, R/2, -d1/2, 1]]).T,
        np.array([[R/2, -R/2, -d1/2, 1]]).T,
        np.array([[-R/2, R/2, -d1/2, 1]]).T,
        np.array([[-R/2, -R/2, -d1/2, 1]]).T,
    )  # 1座標系からみた制御点位置

    r_bars_in_1 = (
        np.array([[0, 0, 0, 1]]).T,
    )

    r_bars_in_2 = (
        np.array([[R/2, R/2, -d1/3, 1]]).T,
        np.array([[R/2, -R/2, -d1/3, 1]]).T,
        np.array([[-R/2, R/2, -d1/3, 1]]).T,
        np.array([[-R/2, -R/2, -d1/3, 1]]).T,
        np.array([[R/2, R/2, -d1*2/3, 1]]).T,
        np.array([[R/2, -R/2, -d1*2/3, 1]]).T,
        np.array([[-R/2, R/2, -d1*2/3, 1]]).T,
        np.array([[-R/2, -R/2, -d1*2/3, 1]]).T,
    )

    r_bars_in_3 = (
        np.array([[0, 0, -R/2, 1]]).T,
        np.array([[0, 0, R/2, 1]]).T,
    )

    r_bars_in_4 = (
        np.array([[R/2, R/2, -d5*2/3, 1]]).T,
        np.array([[R/2, -R/2, -d5*2/3, 1]]).T,
        np.array([[-R/2, R/2, -d5*2/3, 1]]).T,
        np.array([[-R/2, -R/2, -d5*2/3, 1]]).T,
        np.array([[R/2.5, R/2.5, -d5/4, 1]]).T,
        np.array([[R/2.5, -R/2.5, -d5/4, 1]]).T,
        np.array([[-R/2.5, R/2.5, -d5/4, 1]]).T,
        np.array([[-R/2.5, -R/2.5, -d5/4, 1]]).T,
    )

    r_bars_in_5 = (
        np.array([[0, 0, R/2, 1]]).T,
        np.array([[0, 0, -R/2, 1]]).T,
    )

    r_bars_in_6 = (
        np.array([[0, 0, -R, 1]]).T,
        np.array([[R/2, R/2, R, 1]]).T,
        np.array([[R/2, -R/2, R, 1]]).T,
        np.array([[-R/2, R/2, R, 1]]).T,
        np.array([[-R/2, -R/2, R, 1]]).T,
    )

    r_bars_in_GL = (
        np.array([[0, 0, R/2, 1]]).T,
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
    
    
    ee_id = (7, 0)

    def __init__(self, flame_num, position_num):
        self.htm = self.HTM[flame_num]
        self.jo = self.JO[flame_num]
        self.jrx = self.JRX[flame_num]
        self.jry = self.JRY[flame_num]
        self.jrz = self.JRZ[flame_num]
        self.jo_dot = self.JO_DOT[flame_num]
        self.jrx_dot = self.JRX_DOT[flame_num]
        self.jry_dot = self.JRY_DOT[flame_num]
        self.jrz_dot = self.JRZ_DOT[flame_num]
        self.r_bar = self.R_BARS_ALL[flame_num][position_num]
    
    def phi(self, q):
        return (self.htm(q) @ self.r_bar)[:3, :]
    
    def J(self, q):
        return (self.jrx(q)*self.r_bar[0,0] + self.jry(q)*self.r_bar[1,0] + self.jrz(q)*self.r_bar[2,0] + self.jo(q))

    def J_dot(self, q, dq):
        return (self.jrx_dot(q, dq)*self.r_bar[0,0] + self.jry_dot(q, dq)*self.r_bar[1,0] + self.jrz_dot(q, dq)*self.r_bar[2,0] + self.jo_dot(q, dq))






if __name__ == "__main__":
    pass