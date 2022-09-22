import numpy as np
import numpy as np
from math import cos as cos
from math import sin as sin
import sys
sys.path.append(".")
import mappings
from numba import njit

@njit('f8[:,:](f8, f8, f8, f8, f8', cache=True)
def o(x, y, theta, local_x, local_y):
    R = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)],
    ])
    return R @ np.array([[local_x],[local_y]]) + np.array([[x],[y]])


@njit('f8[:,:](f8, f8, f8, f8, f8, f8, f8)', cache=True)
def j(theta, local_x, local_y, beta, L, v, xi):
    j_phi = np.array([
        [1.0, 0.0, -sin(theta)*local_x - cos(theta)*local_y],
        [0.0, 1.0, cos(theta)*local_x - sin(theta)*local_y]
    ])
    j_ = np.array([
        [cos(theta), 0.0],
        [sin(theta), 0.0],
        [sin(2*beta)/L, 4*v*cos(2*beta)/(3*cos(xi)**2 + 1)],
    ])
        
    return j_phi @ j_



@njit('f8[:,:](f8, f8, f8, f8, f8, f8, f8, f8)', cache=True)
def j_dot(theta, omega, local_x, local_y, beta, L, v, xi):
    j_phi = np.array([
        [0.0, 0.0, -cos(theta)*omega*local_x + sin(theta)*omega*local_y],
        [0.0, 0.0, -sin(theta)*omega*local_x - cos(theta)*omega*local_y]
    ])
    j_ = np.array([
        [-sin(theta)*omega, 0.0],
        [cos(theta)*omega, 0.0],
        [0.0, 4*v*cos(2*beta)/(3*cos(xi)**2 + 1)],
    ])
        
    return j_phi @ j_





def q_neutral():
    return np.array([[0, 0, 0]]).T  # ニュートラルの姿勢

def q_min():
    return np.array([[-90, -90, -90, -90]]).T * np.pi/180

def q_max():
    return np.array([[90, 90, 90, 90]]).T * np.pi/180


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


    ee_id = (4, 0)

    def __init__(self, frame_num, position_num):
        
        self.beta = 0
        self.L = 2

        self.local_x, self.local_y = self.RS_ALL[frame_num][position_num]
    
    def phi(self, q):
        return o(q[0,0], q[1,0], q[2,0], self.local_x, self.local_y)
    
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