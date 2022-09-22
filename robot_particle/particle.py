import numpy as np
import numpy as np
from math import cos as cos
from math import sin as sin

import mappings


def q_neutral():
    return np.array([[0, 0]]).T  # ニュートラルの姿勢

def q_min():
    return np.array([[-1e10, -1e10]]).T

def q_max():
    return -q_min()


class CPoint(mappings.Identity):
    c_dim = 2
    t_dim = 2


    rs_in_0 = (
        (0, 0),
    )  # ジョイント1によって回転する制御点


    # 追加
    RS_ALL = (
        rs_in_0,
    )


    ee_id = (0, 0)

    def __init__(self, frame_num, position_num):
        pass
    
    def phi(self, q):
        return q
    
    def J(self, q):
        return np.eye(2)

    def J_dot(self, q, dq):
        return np.zeros((2, 2))


def JOINT_PHI():
    return (
        lambda q: q,
    )

if __name__ == "__main__":
    pass