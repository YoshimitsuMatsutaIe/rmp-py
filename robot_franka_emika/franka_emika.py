import numpy as np
import mappings


import sys
sys.path.append('.')

from robot_franka_emika.htm import *
from robot_franka_emika.Jos import *
from robot_franka_emika.JRxs import *
from robot_franka_emika.JRys import *
from robot_franka_emika.JRzs import *
from robot_franka_emika.Jo_dots import *
from robot_franka_emika.JRx_dots import *
from robot_franka_emika.JRy_dots import *
from robot_franka_emika.JRz_dots import *


def q_neutral():
    return np.array([[0, 0, 0, 0, 0, 0, 0]]).T * np.pi/180  # ニュートラルの姿勢

def q_min():
    return np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]]).T

def q_max():
    return np.array([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]]).T



class CPoint(mappings.Identity):
    c_dim = 7
    t_dim = 3

    d1 = 0.333
    d3 = 0.316
    d5 = 0.384
    df = 0.107
    a4 = 0.0825
    a5 = -0.0825
    a7 = 0.088


    R = a7
    
    R0 = 108e-3

    rs_in_0 = (
        (R0/2, R0/2, -d1/3),
        (R0/2, -R0/2, -d1/3),
        (-R0/2, R0/2, -d1/3),
        (-R0/2, -R0/2, -d1/3),
        (0, -R0, 0),
        (R0/2, 0, 0),
        (-R0/2, 0, 0),
    )  # ジョイント1によって回転する制御点

    rs_in_1 = (
        (0, 0, R0),
        (R0/2, -d3/4, R0/2),
        (R0/2, -d3/4, -R0/2),
        (-R0/2, -d3/4, R0/2),
        (-R0/2, -d3/4, -R0/2)
    )

    rs_in_2 = (
        (R/2, R/2, -d1/3),
        (R/2, -R/2, -d1/3),
        (-R/2, R/2, -d1/3),
        (-R/2, -R/2, -d1/3),
        # np.array([[R/2, R/2, -d1*2/3, 1]]).T,
        # np.array([[R/2, -R/2, -d1*2/3, 1]]).T,
        # np.array([[-R/2, R/2, -d1*2/3, 1]]).T,
        # np.array([[-R/2, -R/2, -d1*2/3, 1]]).T,
    )

    rs_in_3 = (
        (0, 0, -R/2),
        (0, 0, R/2),
    )

    rs_in_4 = (
        (R/2, R/2, -d5*2/3),
        (R/2, -R/2, -d5*2/3),
        (-R/2, R/2, -d5*2/3),
        (-R/2, -R/2, -d5*2/3),
        (R/2.5, R/2.5, -d5/4),
        (R/2.5, -R/2.5, -d5/4),
        (-R/2.5, R/2.5, -d5/4),
        (-R/2.5, -R/2.5, -d5/4),
    )

    rs_in_5 = (
        (0, 0, R/2),
        (0, 0, -R/2),
    )
    rs_in_6 = (
        (0, 0, -R),
        (R/2, R/2, R),
        (R/2, -R/2, R),
        (-R/2, R/2, R),
        (-R/2, -R/2, R),
        (R/2, R/2, 1.5*R),
        (R/2, -R/2, 1.5*R),
        (-R/2, R/2, 1.5*R),
        (-R/2, -R/2, 1.5*R),
    )

    rs_in_GL = (
        (0, 0, d5/2),  # エンドエフェクタの代表位置
        (0, R0, d5/2-0.03),
        (0, -R0, d5/2-0.03),
        (0.02, 0, d5/2-0.03),
        (-0.02, 0, d5/2-0.03),
        (R/2, 0, d5/2-0.07),
        (-R/2, 0, d5/2-0.07)
    )

    # 追加
    RS_ALL = (
        rs_in_0, rs_in_1, rs_in_2, rs_in_3, rs_in_4, rs_in_5, rs_in_6, rs_in_GL,
    )


    ee_id = (7, 0)

    def __init__(self, frame_num, position_num):
        self.o = lambda q: o(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.rx = lambda q: rx(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.ry = lambda q: ry(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.rz = lambda q: rz(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jo = lambda q: jo(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jrx = lambda q: jrx(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jry = lambda q: jry(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jrz = lambda q: jrz(q, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jo_dot = lambda q, dq: jo_dot(q, dq, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jrx_dot = lambda q, dq: jrx_dot(q, dq, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jry_dot = lambda q, dq: jry_dot(q, dq, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.jrz_dot = lambda q, dq: jo_dot(q, dq, frame_num, self.d1, self.d3, self.d5, self.df, self.a4, self.a5, self.a7)
        self.r = self.RS_ALL[frame_num][position_num]
    
    
    def phi(self, q):
        return self.rx(q)*self.r[0] + self.ry(q)*self.r[1] + self.rz(q)*self.r[2] + self.o(q)
    
    def J(self, q):
        return (self.jrx(q)*self.r[0] + self.jry(q)*self.r[1] + self.jrz(q)*self.r[2] + self.jo(q))

    def J_dot(self, q, dq):
        return (self.jrx_dot(q, dq)*self.r[0] + self.jry_dot(q, dq)*self.r[1] + self.jrz_dot(q, dq)*self.r[2] + self.jo_dot(q, dq))


def JOINT_PHI():
    return (
        lambda q: np.zeros((3,1)),
        lambda q: o(q, 0, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 1, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 2, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 3, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 4, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 5, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 6, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
        lambda q: o(q, 7, CPoint.d1, CPoint.d3, CPoint.d5, CPoint.df, CPoint.a4, CPoint.a5, CPoint.a7),
    )



if __name__ == "__main__":
    pass