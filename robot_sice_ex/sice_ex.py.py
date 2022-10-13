import numpy as np
import numpy as np
from math import cos as c
from math import sin as s
import mappings
from numba import njit

import sys
sys.path.append('.')

import robot_sice_ex.rx as rx
import robot_sice_ex.ry as ry
import robot_sice_ex.o as o
import robot_sice_ex.JRxs as jrx
import robot_sice_ex.JRys as jry
import robot_sice_ex.Jos as jo
import robot_sice_ex.JRx_dots as jrx_dot
import robot_sice_ex.JRy_dots as jry_dot
import robot_sice_ex.Jo_dots as jo_dot


class CPoint(mappings.Identity):

    t_dim = 2

    rs_in_0 = (
        (0, 0),
    )  # ジョイント1によって回転する制御点

    rs_in_1 = (
        (0, 0),
    )

    rs_in_2 = (
        (0, 0),
    )

    rs_in_3 = (
        (0, 0),
    )

    rs_in_GL = (
        (0, 0),
    )

    # 追加
    RS_ALL = (
        rs_in_0, rs_in_1, rs_in_2, rs_in_3, rs_in_GL,
    )


    ee_id = (4, 0)


    def __hoge(q, dq):
        if c_dim >

    def o(self, q):
        

    def __init__(self, frame_num, position_num, c_dim=4, ls=None):
        if ls is None:
            ls = [1 for _ in range(c_dim)]
        
        
        
        self.o = lambda q: o(q, frame_num, *ls)
        self.rx = lambda q: rx(q, frame_num)
        self.ry = lambda q: ry(q, frame_num)
        self.jo = lambda q: jo(q, frame_num, self.l1, self.l2, self.l3, self.l4)
        self.jrx = lambda q: jrx(q, frame_num)
        self.jry = lambda q: jry(q, frame_num)
        self.jo_dot = lambda q, q_dot: jo_dot(q, q_dot, frame_num, self.l1, self.l2, self.l3, self.l4)
        self.jrx_dot = lambda q, q_dot: jrx_dot(q, q_dot, frame_num)
        self.jry_dot = lambda q: jry_dot(q, frame_num)
        self.r = self.RS_ALL[frame_num][position_num]
    
    def phi(self, q):
        return self.rx(q) * self.r[0] + self.ry(q) * self.r[1] + self.o(q)
    
    def J(self, q):
        return (self.jrx(q)*self.r[0] + self.jry(q)*self.r[1] + self.jo(q))

    def J_dot(self, q, dq):
        return self.jrx_dot(q, dq)*self.r[0] + self.jry_dot(q)*self.r[1] + self.jo_dot(q, dq)


def JOINT_PHI(l1=1.0, l2=1.0, l3=1.0, l4=1.0):
    return (
        lambda _: np.zeros((2,1)),
        lambda q: o(q, 0, l1, l2, l3, l4),
        lambda q: o(q, 1, l1, l2, l3, l4),
        lambda q: o(q, 2, l1, l2, l3, l4),
        lambda q: o(q, 3, l1, l2, l3, l4),
        lambda q: o(q, 4, l1, l2, l3, l4),
    )

if __name__ == "__main__":
    pass