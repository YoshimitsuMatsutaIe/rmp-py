"""baxter"""

import numpy as np
import mappings

import sys
sys.path.append('.')

from robot_baxter.htm import *
from robot_baxter.Jos import *
from robot_baxter.JRxs import *
from robot_baxter.JRys import *
from robot_baxter.JRzs import *
from robot_baxter.Jo_dots import *
from robot_baxter.JRx_dots import *
from robot_baxter.JRy_dots import *
from robot_baxter.JRz_dots import *


def q_neutral():
    return np.array([[0, -31, 0, 43, 0, 72, 0]]) * np.pi/180  # ニュートラルの姿勢

def q_min():
    return np.array([[-141, -123, -173, -3, -175, -90, -175]]) * np.pi/180

def q_max():
    return np.array([[51, 60, 173, 150, 175, 120, 175]]) * np.pi/180


class CPoint(mappings.Identity):
    c_dim = 7
    t_dim = 3
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

    # 制御点のローカル座標
    R = 0.05

    rs_in_0 = (
        (0, L1/2, -L0/2),
        (0, -L1/2, -L0/2),
        (L1/2, 0, -L0/2),
        (-L1/2, 0, -L0/2),
    )  # 1座標系からみた制御点位置

    rs_in_1 = (
        (0, 0, L3/2),
        (0, 0, -L3/2),
    )

    rs_in_2 = (
        (0, L3/2, -L2*2/3),
        (0, -L3/2, -L2*2/3),
        (L3/2, 0, -L2*2/3),
        (-L3/2, 0, -L2*2/3),
        (0, L3/2, -L2*1/3),
        (0, -L3/2, -L2*1/3),
        (L3/2, 0, -L2*1/3),
        (-L3/2, 0, -L2*1/3),
    )

    rs_in_3 = (
        (0, 0, L3/2),
        (0, 0, -L3/2),
    )

    rs_in_4 = (
        (0, R/2, -L4/3),
        (0, -R/2, -L4/3),
        (R/2, 0, -L4/3),
        (-R/2, 0, -L4/3),
        (0, R/2, -L4/3*2),
        (0, -R/2, -L4/3*2),
        (R/2, 0, -L4/3*2),
        (-R/2, 0, -L4/3*2),
    )

    rs_in_5 = (
        (0, 0, L5/2),
        (0, 0, -L5/2),
    )

    rs_in_6 = (
        (0, R/2, L6/3),
        (0, -R/2, L6/3),
        (R/2, 0, L6/3),
        (-R/2, 0, L6/3),
        (0, R/2, L6/3*2),
        (0, -R/2, L6/3*2),
        (R/2, 0, L6/3*2),
        (-R/2, 0, L6/3*2),
    )

    rs_in_GL = (
        (0, 0, 0),
    )

    # 追加
    RS_ALL = (
        rs_in_0, rs_in_1, rs_in_2, rs_in_3, rs_in_4, rs_in_5, rs_in_6, rs_in_GL,
    )


    ee_id = (7, 0)
    
    
    def __init__(self, frame_num, position_num):
        
        self.o = lambda q: o(q, frame_num)
        self.rx = lambda q: rx(q, frame_num)
        self.ry = lambda q: ry(q, frame_num)
        self.rz = lambda q: rz(q, frame_num)
        self.jo = lambda q: jo(q, frame_num)
        self.jrx = lambda q: jrx(q, frame_num)
        self.jry = lambda q: jry(q, frame_num)
        self.jrz = lambda q: jrz(q, frame_num)
        self.jo_dot = lambda q, dq: jo_dot(q, dq, frame_num)
        self.jrx_dot = lambda q, dq: jrx_dot(q, dq, frame_num)
        self.jry_dot = lambda q, dq: jry_dot(q, dq, frame_num)
        self.jrz_dot = lambda q, dq: jrz_dot(q, dq, frame_num)
        self.r = self.RS_ALL[frame_num][position_num]
    
    def phi(self, q):
        return self.rx(q)*self.r[0] + self.ry(q)*self.r[1] + self.rz(q)*self.r[2] + self.o(q)
    
    def J(self, q):
        return self.jrx(q)*self.r[0] + self.jry(q)*self.r[1] + self.jrz(q)*self.r[2] + self.jo(q)

    def J_dot(self, q, dq):
        return self.jrx_dot(q, dq)*self.r[0] + self.jry_dot(q, dq)*self.r[1] + self.jrz_dot(q, dq)*self.r[2] + self.jo_dot(q, dq)



def JOINT_PHI():
    return (
        lambda x: np.zeros((3,1)),
        lambda q: o(q, 0),
        lambda q: o(q, 1),
        lambda q: o(q, 2),
        lambda q: o(q, 3),
        lambda q: o(q, 4),
        lambda q: o(q, 5),
        lambda q: o(q, 6),
        lambda q: o(q, 7),
    )


if __name__ == "__main__":
    pass