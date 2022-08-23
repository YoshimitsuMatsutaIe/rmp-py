import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq

import sys
sys.path.append('.')
from sice.params import SiceParam

class JRX_DOT(SiceParam):
    @classmethod
    def jrx_0_dot(cls, q, dq):
        return np.array([[-c(q[0, 0])*dq[0, 0], 0, 0, 0], [-s(q[0, 0])*dq[0, 0], 0, 0, 0]])
    @classmethod
    def jrx_1_dot(cls, q, dq):
        return np.array([[-(dq[0, 0] + dq[1, 0])*c(q[0, 0] + q[1, 0]), -(dq[0, 0] + dq[1, 0])*c(q[0, 0] + q[1, 0]), 0, 0], [-(dq[0, 0] + dq[1, 0])*s(q[0, 0] + q[1, 0]), -(dq[0, 0] + dq[1, 0])*s(q[0, 0] + q[1, 0]), 0, 0]])
    @classmethod
    def jrx_2_dot(cls, q, dq):
        return np.array([[-(dq[0, 0] + dq[1, 0] + dq[2, 0])*c(q[0, 0] + q[1, 0] + q[2, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0])*c(q[0, 0] + q[1, 0] + q[2, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0])*c(q[0, 0] + q[1, 0] + q[2, 0]), 0], [-(dq[0, 0] + dq[1, 0] + dq[2, 0])*s(q[0, 0] + q[1, 0] + q[2, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0])*s(q[0, 0] + q[1, 0] + q[2, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0])*s(q[0, 0] + q[1, 0] + q[2, 0]), 0]])
    @classmethod
    def jrx_3_dot(cls, q, dq):
        return np.array([[s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0], s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0], s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0], s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0]], [-(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0])]])
    @classmethod
    def jrx_ee_dot(cls, q, dq):
        return np.array([[s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0], s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0], s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0], s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0] + q[1, 0])*s(q[3, 0])*c(q[2, 0])*dq[3, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[0, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[1, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[2, 0] + s(q[1, 0] + q[2, 0])*s(q[0, 0])*c(q[3, 0])*dq[3, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] + s(q[1, 0] + q[3, 0])*s(q[2, 0])*c(q[0, 0])*dq[3, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[0, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[2, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*dq[3, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*dq[3, 0]], [-(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -(dq[0, 0] + dq[1, 0] + dq[2, 0] + dq[3, 0])*s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0])]])
