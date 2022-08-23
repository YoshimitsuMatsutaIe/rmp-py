import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq

import sys
sys.path.append('.')
from sice.params import SiceParam

class JRY_DOT(SiceParam):
    @classmethod
    def jry_0_dot(cls, q, dq):
        return np.array([[-c(q[0, 0]), 0, 0, 0], [-s(q[0, 0]), 0, 0, 0]])
    @classmethod
    def jry_1_dot(cls, q, dq):
        return np.array([[-c(q[0, 0] + q[1, 0]), -c(q[0, 0] + q[1, 0]), 0, 0], [-s(q[0, 0] + q[1, 0]), -s(q[0, 0] + q[1, 0]), 0, 0]])
    @classmethod
    def jry_2_dot(cls, q, dq):
        return np.array([[-c(q[0, 0] + q[1, 0] + q[2, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0]), 0], [-s(q[0, 0] + q[1, 0] + q[2, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0]), 0]])
    @classmethod
    def jry_3_dot(cls, q, dq):
        return np.array([[-c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0])], [-s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0])]])
    @classmethod
    def jry_ee_dot(cls, q, dq):
        return np.array([[-c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -c(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0])], [-s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0]), -s(q[0, 0] + q[1, 0] + q[2, 0] + q[3, 0])]])
