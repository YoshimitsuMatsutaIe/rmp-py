import numpy as np
import mappings

import sys
sys.path.append('.')

from sice.htm import HTM
from sice.Jos import JO
from sice.JRxs import JRX
from sice.JRys import JRY
from sice.Jo_dots import JO_DOT
from sice.JRx_dots import JRX_DOT
from sice.JRy_dots import JRY_DOT


def htm_0(q):
    return np.block([
        [HTM.rx_0(q), HTM.ry_0(q), HTM.o_0(q)],
        [np.array([[0, 0, 1]])]
    ])

def htm_1(q):
    return np.block([
        [HTM.rx_1(q), HTM.ry_1(q), HTM.o_1(q)],
        [np.array([[0, 0, 1]])]
    ])

def htm_2(q):
    return np.block([
        [HTM.rx_2(q), HTM.ry_2(q), HTM.o_2(q)],
        [np.array([[0, 0, 1]])]
    ])

def htm_3(q):
    return np.block([
        [HTM.rx_3(q), HTM.ry_3(q), HTM.o_3(q)],
        [np.array([[0, 0, 1]])]
    ])

def htm_ee(q):
    return np.block([
        [HTM.rx_ee(q), HTM.ry_ee(q), HTM.o_ee(q)],
        [np.array([[0, 0, 1]])]
    ])



class CPoint(mappings.Identity):
    q_neutral = np.array([[0, 0, 0, 0]]).T * np.pi/180  # ニュートラルの姿勢
    q_min = np.array([[-90, -90, -90, -90]]).T * np.pi/180
    q_max = np.array([[90, 90, 90, 90]]).T * np.pi/180

    # 制御点のローカル座標

    c_dim = 4
    t_dim = 2

    r_bars_in_0 = (
        np.array([[0, 0, 1]]).T,
    )  # ジョイント1によって回転する制御点

    r_bars_in_1 = (
        np.array([[0, 0, 1]]).T,
    )

    r_bars_in_2 = (
        np.array([[0, 0, 1]]).T,
    )

    r_bars_in_3 = (
        np.array([[0, 0, 1]]).T,
    )

    r_bars_in_GL = (
        np.array([[0, 0, 1]]).T,
    )

    # 追加
    R_BARS_ALL = (
        r_bars_in_0, r_bars_in_1, r_bars_in_2, r_bars_in_3, r_bars_in_GL,
    )

    r_bar_zero = np.array([[0, 0, 0, 1]]).T

    JOINT_PHI = (lambda x: np.zeros((2,1)), HTM.o_0, HTM.o_1, HTM.o_2, HTM.o_3, HTM.o_ee)
    HTM = (htm_0, htm_1, htm_2, htm_3, htm_ee)
    JO = (JO.jo_0, JO.jo_1, JO.jo_2, JO.jo_3, JO.jo_ee)
    JRX = (JRX.jrx_0, JRX.jrx_1, JRX.jrx_2, JRX.jrx_3, JRX.jrx_ee)
    JRY = (JRY.jry_0, JRY.jry_1, JRY.jry_2, JRY.jry_3, JRY.jry_ee)
    JO_DOT = (JO_DOT.jo_0_dot, JO_DOT.jo_1_dot, JO_DOT.jo_2_dot, JO_DOT.jo_3_dot, JO_DOT.jo_ee_dot)
    JRX_DOT = (JRX_DOT.jrx_0_dot, JRX_DOT.jrx_1_dot, JRX_DOT.jrx_2_dot, JRX_DOT.jrx_3_dot, JRX_DOT.jrx_ee_dot)
    JRY_DOT = (JRY_DOT.jry_0_dot, JRY_DOT.jry_1_dot, JRY_DOT.jry_2_dot, JRY_DOT.jry_3_dot, JRY_DOT.jry_ee_dot)
    
    
    ee_id = (4, 0)

    def __init__(self, flame_num, position_num):
        self.htm = self.HTM[flame_num]
        self.jo = self.JO[flame_num]
        self.jrx = self.JRX[flame_num]
        self.jry = self.JRY[flame_num]
        self.jo_dot = self.JO_DOT[flame_num]
        self.jrx_dot = self.JRX_DOT[flame_num]
        self.jry_dot = self.JRY_DOT[flame_num]
        self.r_bar = self.R_BARS_ALL[flame_num][position_num]
    
    def phi(self, q):
        return (self.htm(q) @ self.r_bar)[:2, :]
    
    def J(self, q):
        return (self.jrx(q)*self.r_bar[0,0] + self.jry(q)*self.r_bar[1,0] + self.jo(q))

    def J_dot(self, q, dq):
        return (self.jrx_dot(q, dq)*self.r_bar[0,0] + self.jry_dot(q, dq)*self.r_bar[1,0] + self.jo_dot(q, dq))




if __name__ == "__main__":
    pass