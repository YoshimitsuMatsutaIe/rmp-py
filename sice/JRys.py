import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq

import sys
sys.path.append('.')
from sice.params import SiceParam

class JRY(SiceParam):
    @classmethod
    def jry_0(cls, q):
        return np.array([[-c(q[0, 0]), 0, 0, 0], [-s(q[0, 0]), 0, 0, 0]])
    @classmethod
    def jry_1(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0]) - c(q[0, 0])*c(q[1, 0]), s(q[0, 0])*s(q[1, 0]) - c(q[0, 0])*c(q[1, 0]), 0, 0], [-s(q[0, 0])*c(q[1, 0]) - s(q[1, 0])*c(q[0, 0]), -s(q[0, 0])*c(q[1, 0]) - s(q[1, 0])*c(q[0, 0]), 0, 0]])
    @classmethod
    def jry_2(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), 0], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), 0]])
    @classmethod
    def jry_3(cls, q):
        return np.array([[-s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])]])
    @classmethod
    def jry_ee(cls, q):
        return np.array([[-s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])]])
