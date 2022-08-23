import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq

from params import SiceParam

class HTM(SiceParam):
    @classmethod
    def o_0(cls, q):
        return np.array([[0], [0]])
    @classmethod
    def o_1(cls, q):
        return np.array([[cls.l1*c(q[0, 0])], [cls.l1*s(q[0, 0])]])
    @classmethod
    def o_2(cls, q):
        return np.array([[cls.l1*c(q[0, 0]) - cls.l2*s(q[0, 0])*s(q[1, 0]) + cls.l2*c(q[0, 0])*c(q[1, 0])], [cls.l1*s(q[0, 0]) + cls.l2*s(q[0, 0])*c(q[1, 0]) + cls.l2*s(q[1, 0])*c(q[0, 0])]])
    @classmethod
    def o_3(cls, q):
        return np.array([[cls.l1*c(q[0, 0]) - cls.l2*s(q[0, 0])*s(q[1, 0]) + cls.l2*c(q[0, 0])*c(q[1, 0]) - cls.l3*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - cls.l3*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) - cls.l3*s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + cls.l3*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [cls.l1*s(q[0, 0]) + cls.l2*s(q[0, 0])*c(q[1, 0]) + cls.l2*s(q[1, 0])*c(q[0, 0]) - cls.l3*s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + cls.l3*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + cls.l3*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + cls.l3*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])]])
    @classmethod
    def o_ee(cls, q):
        return np.array([[cls.l1*c(q[0, 0]) - cls.l2*s(q[0, 0])*s(q[1, 0]) + cls.l2*c(q[0, 0])*c(q[1, 0]) - cls.l3*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - cls.l3*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) - cls.l3*s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + cls.l3*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + cls.l4*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) - cls.l4*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - cls.l4*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) - cls.l4*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - cls.l4*s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - cls.l4*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) - cls.l4*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + cls.l4*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])], [cls.l1*s(q[0, 0]) + cls.l2*s(q[0, 0])*c(q[1, 0]) + cls.l2*s(q[1, 0])*c(q[0, 0]) - cls.l3*s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + cls.l3*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + cls.l3*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + cls.l3*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]) - cls.l4*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - cls.l4*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) - cls.l4*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) + cls.l4*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - cls.l4*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) + cls.l4*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + cls.l4*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + cls.l4*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])]])
    @classmethod
    def rx_0(cls, q):
        return np.array([[c(q[0, 0])], [s(q[0, 0])]])
    @classmethod
    def rx_1(cls, q):
        return np.array([[-s(q[0, 0])*s(q[1, 0]) + c(q[0, 0])*c(q[1, 0])], [s(q[0, 0])*c(q[1, 0]) + s(q[1, 0])*c(q[0, 0])]])
    @classmethod
    def rx_2(cls, q):
        return np.array([[-s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) - s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [-s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])]])
    @classmethod
    def rx_3(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) - s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])], [-s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) - s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])]])
    @classmethod
    def rx_ee(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) - s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])], [-s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) - s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])]])
    @classmethod
    def ry_0(cls, q):
        return np.array([[-s(q[0, 0])], [c(q[0, 0])]])
    @classmethod
    def ry_1(cls, q):
        return np.array([[-s(q[0, 0])*c(q[1, 0]) - s(q[1, 0])*c(q[0, 0])], [-s(q[0, 0])*s(q[1, 0]) + c(q[0, 0])*c(q[1, 0])]])
    @classmethod
    def ry_2(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])], [-s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) - s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])]])
    @classmethod
    def ry_3(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) - s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])]])
    @classmethod
    def ry_ee(cls, q):
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) + s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) - s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) - s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])]])




if __name__ == "__main__":
    q = np.array([[1, 0]]).T
    print(htm.o_1(q))