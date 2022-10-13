import numpy as np
import numpy as np
from math import cos as c
from math import sin as s
import mappings
from numba import njit

import sys
sys.path.append('.')

from robot_sice_ex.rx import *
from robot_sice_ex.ry import *
from robot_sice_ex.o import *
from robot_sice_ex.JRxs import *
from robot_sice_ex.JRys import *
from robot_sice_ex.Jos import *
from robot_sice_ex.JRx_dots import *
from robot_sice_ex.JRy_dots import *
from robot_sice_ex.Jo_dots import *

c_dim = 4
def q_neutral():
    return np.array([[0 for _ in range(c_dim)]]).T * np.pi/180  # ニュートラルの姿勢

def q_min():
    return np.array([[-90 for _ in range(c_dim)]]).T * np.pi/180

def q_max():
    return -q_min()


class CPoint(mappings.Identity):

    t_dim = 2
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

    def __init__(self, frame_num, position_num, ls=None):
        self.frame_num = frame_num
        if ls is None:
            self.ls = [1 for _ in range(c_dim)]
        else:
            assert len(ls) != c_dim
            self.ls = ls
        
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
    
    
    def __args_unpack(self, q, dq=None):
        args = np.ravel(q).tolist() + [0 for _ in range(10 - self.c_dim)]
        if dq is None:
            args += [0 for _ in range(10)]
        else:
            args += np.ravel(dq).tolist() + [0 for _ in range(10 - self.c_dim)]
        args += self.ls + [0 for _ in range(10 - self.c_dim)]
        return args
    
    def o(self, q):
        args = self.__args_unpack(q)
        if self.frame_num == 0:   return o_0(*args)
        elif self.frame_num == 1: return o_1(*args)
        elif self.frame_num == 2: return o_2(*args)
        elif self.frame_num == 3: return o_3(*args)
        elif self.frame_num == 4: return o_4(*args)
        elif self.frame_num == 5: return o_5(*args)
        elif self.frame_num == 6: return o_6(*args)
        elif self.frame_num == 7: return o_7(*args)
        # elif self.frame_num == 8: return o_8(*args)
        # elif self.frame_num == 9: return o_9(*args)
        else:
            assert False
    
    def rx(self, q):
        args = self.__args_unpack(q)
        if self.frame_num == 0:   return rx_0(*args)
        elif self.frame_num == 1: return rx_1(*args)
        elif self.frame_num == 2: return rx_2(*args)
        elif self.frame_num == 3: return rx_3(*args)
        elif self.frame_num == 4: return rx_4(*args)
        elif self.frame_num == 5: return rx_5(*args)
        elif self.frame_num == 6: return rx_6(*args)
        elif self.frame_num == 7: return rx_7(*args)
        # elif self.frame_num == 8: return rx_8(*args)
        # elif self.frame_num == 9: return rx_9(*args)
        else:
            assert False
    
    def ry(self, q):
        args = self.__args_unpack(q)
        if self.frame_num == 0:   return ry_0(*args)
        elif self.frame_num == 1: return ry_1(*args)
        elif self.frame_num == 2: return ry_2(*args)
        elif self.frame_num == 3: return ry_3(*args)
        elif self.frame_num == 4: return ry_4(*args)
        elif self.frame_num == 5: return ry_5(*args)
        elif self.frame_num == 6: return ry_6(*args)
        elif self.frame_num == 7: return ry_7(*args)
        # elif self.frame_num == 8: return ry_8(*args)
        # elif self.frame_num == 9: return ry_9(*args)
        else:
            assert False
    
    def jo(self, q):
        args = self.__args_unpack(q)
        if self.frame_num == 0:   return jo_0(*args)
        elif self.frame_num == 1: return jo_1(*args)
        elif self.frame_num == 2: return jo_2(*args)
        elif self.frame_num == 3: return jo_3(*args)
        elif self.frame_num == 4: return jo_4(*args)
        elif self.frame_num == 5: return jo_5(*args)
        elif self.frame_num == 6: return jo_6(*args)
        elif self.frame_num == 7: return jo_7(*args)
        # elif self.frame_num == 8: return jo_8(*args)
        # elif self.frame_num == 9: return jo_9(*args)
        else:
            assert False
    
    def jrx(self, q):
        args = self.__args_unpack(q)
        if self.frame_num == 0:   return jrx_0(*args)
        elif self.frame_num == 1: return jrx_1(*args)
        elif self.frame_num == 2: return jrx_2(*args)
        elif self.frame_num == 3: return jrx_3(*args)
        elif self.frame_num == 4: return jrx_4(*args)
        elif self.frame_num == 5: return jrx_5(*args)
        elif self.frame_num == 6: return jrx_6(*args)
        elif self.frame_num == 7: return jrx_7(*args)
        # elif self.frame_num == 8: return jrx_8(*args)
        # elif self.frame_num == 9: return jrx_9(*args)
        else:
            assert False
    
    def jry(self, q):
        args = self.__args_unpack(q)
        if self.frame_num == 0:   return jry_0(*args)
        elif self.frame_num == 1: return jry_1(*args)
        elif self.frame_num == 2: return jry_2(*args)
        elif self.frame_num == 3: return jry_3(*args)
        elif self.frame_num == 4: return jry_4(*args)
        elif self.frame_num == 5: return jry_5(*args)
        elif self.frame_num == 6: return jry_6(*args)
        elif self.frame_num == 7: return jry_7(*args)
        # elif self.frame_num == 8: return jry_8(*args)
        # elif self.frame_num == 9: return jry_9(*args)
        else:
            assert False
    
    def jo_dot(self, q, dq):
        args = self.__args_unpack(q, dq)
        if self.frame_num == 0:   return jo_0_dot(*args)
        elif self.frame_num == 1: return jo_1_dot(*args)
        elif self.frame_num == 2: return jo_2_dot(*args)
        elif self.frame_num == 3: return jo_3_dot(*args)
        elif self.frame_num == 4: return jo_4_dot(*args)
        elif self.frame_num == 5: return jo_5_dot(*args)
        elif self.frame_num == 6: return jo_6_dot(*args)
        elif self.frame_num == 7: return jo_7_dot(*args)
        # elif self.frame_num == 8: return jo_8_dot(*args)
        # elif self.frame_num == 9: return jo_9_dot(*args)
        else:
            assert False
    
    def jrx_dot(self, q, dq):
        args = self.__args_unpack(q, dq)
        if self.frame_num == 0:   return jrx_0_dot(*args)
        elif self.frame_num == 1: return jrx_1_dot(*args)
        elif self.frame_num == 2: return jrx_2_dot(*args)
        elif self.frame_num == 3: return jrx_3_dot(*args)
        elif self.frame_num == 4: return jrx_4_dot(*args)
        elif self.frame_num == 5: return jrx_5_dot(*args)
        elif self.frame_num == 6: return jrx_6_dot(*args)
        elif self.frame_num == 7: return jrx_7_dot(*args)
        # elif self.frame_num == 8: return jrx_8_dot(*args)
        # elif self.frame_num == 9: return jrx_9_dot(*args)
        else:
            assert False
    
    def jry_dot(self, q, dq):
        args = self.__args_unpack(q, dq)
        if self.frame_num == 0:   return jry_0_dot(*args)
        elif self.frame_num == 1: return jry_1_dot(*args)
        elif self.frame_num == 2: return jry_2_dot(*args)
        elif self.frame_num == 3: return jry_3_dot(*args)
        elif self.frame_num == 4: return jry_4_dot(*args)
        elif self.frame_num == 5: return jry_5_dot(*args)
        elif self.frame_num == 6: return jry_6_dot(*args)
        elif self.frame_num == 7: return jry_7_dot(*args)
        # elif self.frame_num == 8: return jry_8_dot(*args)
        # elif self.frame_num == 9: return jry_9_dot(*args)
        else:
            assert False
    
    def phi(self, q):
        return self.rx(q) * self.r[0] + self.ry(q) * self.r[1] + self.o(q)
    
    def J(self, q):
        return (self.jrx(q)*self.r[0] + self.jry(q)*self.r[1] + self.jo(q))[:, :self.c_dim]

    def J_dot(self, q, dq):
        print(self.jrx_dot(q, dq).shape)
        print(self.jry_dot(q, dq).shape)
        print(self.jo_dot(q, dq).shape)
        return (self.jrx_dot(q, dq)*self.r[0] + self.jry_dot(q, dq)*self.r[1] + self.jo_dot(q, dq))[:, :self.c_dim]


    def JOINT_PHI_(self):
        return [
            lambda q: o_0(*self.__args_unpack(q)),
            lambda q: o_1(*self.__args_unpack(q)),
            lambda q: o_2(*self.__args_unpack(q)),
            lambda q: o_3(*self.__args_unpack(q)),
            lambda q: o_4(*self.__args_unpack(q)),
            lambda q: o_5(*self.__args_unpack(q)),
            lambda q: o_6(*self.__args_unpack(q)),
            lambda q: o_7(*self.__args_unpack(q)),
            # lambda q: o_8(*self.__args_unpack(q)),
            # lambda q: o_9(*self.__args_unpack(q)),
        ]


def JOINT_PHI(ls=None):
    h = CPoint(0, 0, ls)
    return h.JOINT_PHI_()[:c_dim]

if __name__ == "__main__":
    pass