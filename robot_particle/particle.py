import numpy as np
import numpy as np
from math import cos as cos
from math import sin as sin

import mappings


class CPoint(mappings.Identity):
    rs_in_0 = (
        (0, 0),
    )  # ジョイント1によって回転する制御点


    # 追加
    RS_ALL = (
        rs_in_0,
    )


    ee_id = (0, 0)

    def __init__(self, frame_num, position_num, **kwargs):
        self.c_dim = kwargs.pop('c_dim')
        self.t_dim = self.c_dim
        self.q_neutral =  np.zeros((self.c_dim, 1))  # ニュートラルの姿勢
        self.q_min = np.array([[-1e10] * self.c_dim]).T
        self.q_max = -self.q_min

    
    def phi(self, q):
        return q
    
    def J(self, q):
        return np.eye(self.c_dim)

    def J_dot(self, q, dq):
        return np.zeros((self.c_dim, self.c_dim))

    def calc_all(self, q, dq):
        x = self.phi(q)
        J = self.J(q)
        x_dot = self.velocity(J, dq)
        J_dot = self.J_dot(q, dq)
        return x, x_dot, J, J_dot

    def calc_joint_position_all(self, q):
        return [
            q,
        ]

if __name__ == "__main__":
    pass