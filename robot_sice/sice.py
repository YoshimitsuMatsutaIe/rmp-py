import numpy as np
import numpy as np
from math import cos as c
from math import sin as s
import mappings
from numba import njit

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8))", cache=True)
def o(q, i, l1, l2, l3, l4):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[0.], [0.]])
    elif i == 1:
        return np.array([[l1*c(q0)], [l1*s(q0)]])
    elif i == 2:
        return np.array([[l1*c(q0) - l2*s(q0)*s(q1) + l2*c(q0)*c(q1)], [l1*s(q0) + l2*s(q0)*c(q1) + l2*s(q1)*c(q0)]])
    elif i == 3:
        return np.array([[l1*c(q0) - l2*s(q0)*s(q1) + l2*c(q0)*c(q1) - l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2)], [l1*s(q0) + l2*s(q0)*c(q1) + l2*s(q1)*c(q0) - l3*s(q0)*s(q1)*s(q2) + l3*s(q0)*c(q1)*c(q2) + l3*s(q1)*c(q0)*c(q2) + l3*s(q2)*c(q0)*c(q1)]])
    elif i == 4:
        return np.array([[l1*c(q0) - l2*s(q0)*s(q1) + l2*c(q0)*c(q1) - l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2) + l4*s(q0)*s(q1)*s(q2)*s(q3) - l4*s(q0)*s(q1)*c(q2)*c(q3) - l4*s(q0)*s(q2)*c(q1)*c(q3) - l4*s(q0)*s(q3)*c(q1)*c(q2) - l4*s(q1)*s(q2)*c(q0)*c(q3) - l4*s(q1)*s(q3)*c(q0)*c(q2) - l4*s(q2)*s(q3)*c(q0)*c(q1) + l4*c(q0)*c(q1)*c(q2)*c(q3)], [l1*s(q0) + l2*s(q0)*c(q1) + l2*s(q1)*c(q0) - l3*s(q0)*s(q1)*s(q2) + l3*s(q0)*c(q1)*c(q2) + l3*s(q1)*c(q0)*c(q2) + l3*s(q2)*c(q0)*c(q1) - l4*s(q0)*s(q1)*s(q2)*c(q3) - l4*s(q0)*s(q1)*s(q3)*c(q2) - l4*s(q0)*s(q2)*s(q3)*c(q1) + l4*s(q0)*c(q1)*c(q2)*c(q3) - l4*s(q1)*s(q2)*s(q3)*c(q0) + l4*s(q1)*c(q0)*c(q2)*c(q3) + l4*s(q2)*c(q0)*c(q1)*c(q3) + l4*s(q3)*c(q0)*c(q1)*c(q2)]])
    else:
        assert False


@njit("(f8[:, :](f8[:, :], i8))", cache=True)
def rx(q, i):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[c(q0)], [s(q0)]])
    elif i == 1:
        return np.array([[-s(q0)*s(q1) + c(q0)*c(q1)], [s(q0)*c(q1) + s(q1)*c(q0)]])
    elif i == 2:
        return np.array([[-s(q0)*s(q1)*c(q2) - s(q0)*s(q2)*c(q1) - s(q1)*s(q2)*c(q0) + c(q0)*c(q1)*c(q2)], [-s(q0)*s(q1)*s(q2) + s(q0)*c(q1)*c(q2) + s(q1)*c(q0)*c(q2) + s(q2)*c(q0)*c(q1)]])
    elif i == 3:
        return np.array([[s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)], [-s(q0)*s(q1)*s(q2)*c(q3) - s(q0)*s(q1)*s(q3)*c(q2) - s(q0)*s(q2)*s(q3)*c(q1) + s(q0)*c(q1)*c(q2)*c(q3) - s(q1)*s(q2)*s(q3)*c(q0) + s(q1)*c(q0)*c(q2)*c(q3) + s(q2)*c(q0)*c(q1)*c(q3) + s(q3)*c(q0)*c(q1)*c(q2)]])
    elif i == 4:
        return np.array([[s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)], [-s(q0)*s(q1)*s(q2)*c(q3) - s(q0)*s(q1)*s(q3)*c(q2) - s(q0)*s(q2)*s(q3)*c(q1) + s(q0)*c(q1)*c(q2)*c(q3) - s(q1)*s(q2)*s(q3)*c(q0) + s(q1)*c(q0)*c(q2)*c(q3) + s(q2)*c(q0)*c(q1)*c(q3) + s(q3)*c(q0)*c(q1)*c(q2)]])
    else:
        assert False


@njit("(f8[:, :](f8[:, :], i8))", cache=True)
def ry(q, i):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[-s(q0)], [c(q0)]])
    elif i == 1:
        return np.array([[-s(q0)*c(q1) - s(q1)*c(q0)], [-s(q0)*s(q1) + c(q0)*c(q1)]])
    elif i == 2:
        return np.array([[s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1)], [-s(q0)*s(q1)*c(q2) - s(q0)*s(q2)*c(q1) - s(q1)*s(q2)*c(q0) + c(q0)*c(q1)*c(q2)]])
    elif i == 3:
        return np.array([[s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)], [s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)]])
    elif i == 4:
        return np.array([[s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)], [s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8))", cache=True)
def jo(q, i, l1, l2, l3, l4):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]])
    elif i == 1:
        return np.array([[-l1*s(q0), 0., 0., 0.], [l1*c(q0), 0., 0., 0.]])
    elif i == 2:
        return np.array([[-l1*s(q0) - l2*s(q0)*c(q1) - l2*s(q1)*c(q0), -l2*s(q0)*c(q1) - l2*s(q1)*c(q0), 0., 0.], [l1*c(q0) - l2*s(q0)*s(q1) + l2*c(q0)*c(q1), -l2*s(q0)*s(q1) + l2*c(q0)*c(q1), 0., 0.]])
    elif i == 3:
        return np.array([[-l1*s(q0) - l2*s(q0)*c(q1) - l2*s(q1)*c(q0) + l3*s(q0)*s(q1)*s(q2) - l3*s(q0)*c(q1)*c(q2) - l3*s(q1)*c(q0)*c(q2) - l3*s(q2)*c(q0)*c(q1), -l2*s(q0)*c(q1) - l2*s(q1)*c(q0) + l3*s(q0)*s(q1)*s(q2) - l3*s(q0)*c(q1)*c(q2) - l3*s(q1)*c(q0)*c(q2) - l3*s(q2)*c(q0)*c(q1), l3*s(q0)*s(q1)*s(q2) - l3*s(q0)*c(q1)*c(q2) - l3*s(q1)*c(q0)*c(q2) - l3*s(q2)*c(q0)*c(q1), 0], [l1*c(q0) - l2*s(q0)*s(q1) + l2*c(q0)*c(q1) - l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2), -l2*s(q0)*s(q1) + l2*c(q0)*c(q1) - l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2), -l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2), 0.]])
    elif i == 4:
        return np.array([[-l1*s(q0) - l2*s(q0)*c(q1) - l2*s(q1)*c(q0) + l3*s(q0)*s(q1)*s(q2) - l3*s(q0)*c(q1)*c(q2) - l3*s(q1)*c(q0)*c(q2) - l3*s(q2)*c(q0)*c(q1) + l4*s(q0)*s(q1)*s(q2)*c(q3) + l4*s(q0)*s(q1)*s(q3)*c(q2) + l4*s(q0)*s(q2)*s(q3)*c(q1) - l4*s(q0)*c(q1)*c(q2)*c(q3) + l4*s(q1)*s(q2)*s(q3)*c(q0) - l4*s(q1)*c(q0)*c(q2)*c(q3) - l4*s(q2)*c(q0)*c(q1)*c(q3) - l4*s(q3)*c(q0)*c(q1)*c(q2), -l2*s(q0)*c(q1) - l2*s(q1)*c(q0) + l3*s(q0)*s(q1)*s(q2) - l3*s(q0)*c(q1)*c(q2) - l3*s(q1)*c(q0)*c(q2) - l3*s(q2)*c(q0)*c(q1) + l4*s(q0)*s(q1)*s(q2)*c(q3) + l4*s(q0)*s(q1)*s(q3)*c(q2) + l4*s(q0)*s(q2)*s(q3)*c(q1) - l4*s(q0)*c(q1)*c(q2)*c(q3) + l4*s(q1)*s(q2)*s(q3)*c(q0) - l4*s(q1)*c(q0)*c(q2)*c(q3) - l4*s(q2)*c(q0)*c(q1)*c(q3) - l4*s(q3)*c(q0)*c(q1)*c(q2), l3*s(q0)*s(q1)*s(q2) - l3*s(q0)*c(q1)*c(q2) - l3*s(q1)*c(q0)*c(q2) - l3*s(q2)*c(q0)*c(q1) + l4*s(q0)*s(q1)*s(q2)*c(q3) + l4*s(q0)*s(q1)*s(q3)*c(q2) + l4*s(q0)*s(q2)*s(q3)*c(q1) - l4*s(q0)*c(q1)*c(q2)*c(q3) + l4*s(q1)*s(q2)*s(q3)*c(q0) - l4*s(q1)*c(q0)*c(q2)*c(q3) - l4*s(q2)*c(q0)*c(q1)*c(q3) - l4*s(q3)*c(q0)*c(q1)*c(q2), l4*s(q0)*s(q1)*s(q2)*c(q3) + l4*s(q0)*s(q1)*s(q3)*c(q2) + l4*s(q0)*s(q2)*s(q3)*c(q1) - l4*s(q0)*c(q1)*c(q2)*c(q3) + l4*s(q1)*s(q2)*s(q3)*c(q0) - l4*s(q1)*c(q0)*c(q2)*c(q3) - l4*s(q2)*c(q0)*c(q1)*c(q3) - l4*s(q3)*c(q0)*c(q1)*c(q2)], [l1*c(q0) - l2*s(q0)*s(q1) + l2*c(q0)*c(q1) - l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2) + l4*s(q0)*s(q1)*s(q2)*s(q3) - l4*s(q0)*s(q1)*c(q2)*c(q3) - l4*s(q0)*s(q2)*c(q1)*c(q3) - l4*s(q0)*s(q3)*c(q1)*c(q2) - l4*s(q1)*s(q2)*c(q0)*c(q3) - l4*s(q1)*s(q3)*c(q0)*c(q2) - l4*s(q2)*s(q3)*c(q0)*c(q1) + l4*c(q0)*c(q1)*c(q2)*c(q3), -l2*s(q0)*s(q1) + l2*c(q0)*c(q1) - l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2) + l4*s(q0)*s(q1)*s(q2)*s(q3) - l4*s(q0)*s(q1)*c(q2)*c(q3) - l4*s(q0)*s(q2)*c(q1)*c(q3) - l4*s(q0)*s(q3)*c(q1)*c(q2) - l4*s(q1)*s(q2)*c(q0)*c(q3) - l4*s(q1)*s(q3)*c(q0)*c(q2) - l4*s(q2)*s(q3)*c(q0)*c(q1) + l4*c(q0)*c(q1)*c(q2)*c(q3), -l3*s(q0)*s(q1)*c(q2) - l3*s(q0)*s(q2)*c(q1) - l3*s(q1)*s(q2)*c(q0) + l3*c(q0)*c(q1)*c(q2) + l4*s(q0)*s(q1)*s(q2)*s(q3) - l4*s(q0)*s(q1)*c(q2)*c(q3) - l4*s(q0)*s(q2)*c(q1)*c(q3) - l4*s(q0)*s(q3)*c(q1)*c(q2) - l4*s(q1)*s(q2)*c(q0)*c(q3) - l4*s(q1)*s(q3)*c(q0)*c(q2) - l4*s(q2)*s(q3)*c(q0)*c(q1) + l4*c(q0)*c(q1)*c(q2)*c(q3), l4*s(q0)*s(q1)*s(q2)*s(q3) - l4*s(q0)*s(q1)*c(q2)*c(q3) - l4*s(q0)*s(q2)*c(q1)*c(q3) - l4*s(q0)*s(q3)*c(q1)*c(q2) - l4*s(q1)*s(q2)*c(q0)*c(q3) - l4*s(q1)*s(q3)*c(q0)*c(q2) - l4*s(q2)*s(q3)*c(q0)*c(q1) + l4*c(q0)*c(q1)*c(q2)*c(q3)]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], f8[:,:], i8, f8, f8, f8, f8))", cache=True)
def jo_dot(q, q_dot, i, l1, l2, l3, l4):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    dq0 = q_dot[0,0]
    dq1 = q_dot[1,0]
    dq2 = q_dot[2,0]
    dq3 = q_dot[3,0]
    if i == 0:
        return np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]])
    elif i == 1:
        return np.array([[-l1*c(q0)*dq0, 0., 0., 0.], [-l1*s(q0)*dq0, 0., 0., 0.]])
    elif i == 2:
        return np.array([[-l1*c(q0)*dq0 - l2*c(q0 + q1)*dq0 - l2*c(q0 + q1)*dq1, -l2*(dq0 + dq1)*c(q0 + q1), 0., 0.], [-l1*s(q0)*dq0 - l2*s(q0 + q1)*dq0 - l2*s(q0 + q1)*dq1, -l2*(dq0 + dq1)*s(q0 + q1), 0., 0.]])
    elif i == 3:
        return np.array([[-l1*c(q0)*dq0 - l2*c(q0 + q1)*dq0 - l2*c(q0 + q1)*dq1 - l3*c(q0 + q1 + q2)*dq0 - l3*c(q0 + q1 + q2)*dq1 - l3*c(q0 + q1 + q2)*dq2, -l2*c(q0 + q1)*dq0 - l2*c(q0 + q1)*dq1 - l3*c(q0 + q1 + q2)*dq0 - l3*c(q0 + q1 + q2)*dq1 - l3*c(q0 + q1 + q2)*dq2, -l3*(dq0 + dq1 + dq2)*c(q0 + q1 + q2), 0], [-l1*s(q0)*dq0 - l2*s(q0 + q1)*dq0 - l2*s(q0 + q1)*dq1 - l3*s(q0 + q1 + q2)*dq0 - l3*s(q0 + q1 + q2)*dq1 - l3*s(q0 + q1 + q2)*dq2, -l2*s(q0 + q1)*dq0 - l2*s(q0 + q1)*dq1 - l3*s(q0 + q1 + q2)*dq0 - l3*s(q0 + q1 + q2)*dq1 - l3*s(q0 + q1 + q2)*dq2, -l3*(dq0 + dq1 + dq2)*s(q0 + q1 + q2), 0.]])
    elif i == 4:
        return np.array([[-l1*c(q0)*dq0 - l2*c(q0 + q1)*dq0 - l2*c(q0 + q1)*dq1 - l3*c(q0 + q1 + q2)*dq0 - l3*c(q0 + q1 + q2)*dq1 - l3*c(q0 + q1 + q2)*dq2 + l4*s(q0 + q1)*s(q3)*c(q2)*dq0 + l4*s(q0 + q1)*s(q3)*c(q2)*dq1 + l4*s(q0 + q1)*s(q3)*c(q2)*dq2 + l4*s(q0 + q1)*s(q3)*c(q2)*dq3 + l4*s(q1 + q2)*s(q0)*c(q3)*dq0 + l4*s(q1 + q2)*s(q0)*c(q3)*dq1 + l4*s(q1 + q2)*s(q0)*c(q3)*dq2 + l4*s(q1 + q2)*s(q0)*c(q3)*dq3 + l4*s(q1 + q3)*s(q2)*c(q0)*dq0 + l4*s(q1 + q3)*s(q2)*c(q0)*dq1 + l4*s(q1 + q3)*s(q2)*c(q0)*dq2 + l4*s(q1 + q3)*s(q2)*c(q0)*dq3 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq0 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq1 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq2 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq3 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq0 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq1 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq2 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq3, -l2*c(q0 + q1)*dq0 - l2*c(q0 + q1)*dq1 - l3*c(q0 + q1 + q2)*dq0 - l3*c(q0 + q1 + q2)*dq1 - l3*c(q0 + q1 + q2)*dq2 + l4*s(q0 + q1)*s(q3)*c(q2)*dq0 + l4*s(q0 + q1)*s(q3)*c(q2)*dq1 + l4*s(q0 + q1)*s(q3)*c(q2)*dq2 + l4*s(q0 + q1)*s(q3)*c(q2)*dq3 + l4*s(q1 + q2)*s(q0)*c(q3)*dq0 + l4*s(q1 + q2)*s(q0)*c(q3)*dq1 + l4*s(q1 + q2)*s(q0)*c(q3)*dq2 + l4*s(q1 + q2)*s(q0)*c(q3)*dq3 + l4*s(q1 + q3)*s(q2)*c(q0)*dq0 + l4*s(q1 + q3)*s(q2)*c(q0)*dq1 + l4*s(q1 + q3)*s(q2)*c(q0)*dq2 + l4*s(q1 + q3)*s(q2)*c(q0)*dq3 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq0 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq1 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq2 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq3 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq0 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq1 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq2 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq3, -l3*c(q0 + q1 + q2)*dq0 - l3*c(q0 + q1 + q2)*dq1 - l3*c(q0 + q1 + q2)*dq2 + l4*s(q0 + q1)*s(q3)*c(q2)*dq0 + l4*s(q0 + q1)*s(q3)*c(q2)*dq1 + l4*s(q0 + q1)*s(q3)*c(q2)*dq2 + l4*s(q0 + q1)*s(q3)*c(q2)*dq3 + l4*s(q1 + q2)*s(q0)*c(q3)*dq0 + l4*s(q1 + q2)*s(q0)*c(q3)*dq1 + l4*s(q1 + q2)*s(q0)*c(q3)*dq2 + l4*s(q1 + q2)*s(q0)*c(q3)*dq3 + l4*s(q1 + q3)*s(q2)*c(q0)*dq0 + l4*s(q1 + q3)*s(q2)*c(q0)*dq1 + l4*s(q1 + q3)*s(q2)*c(q0)*dq2 + l4*s(q1 + q3)*s(q2)*c(q0)*dq3 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq0 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq1 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq2 - l4*s(q0)*s(q1)*s(q2)*s(q3)*dq3 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq0 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq1 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq2 - l4*c(q0)*c(q1)*c(q2)*c(q3)*dq3, l4*(s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3)], [-l1*s(q0)*dq0 - l2*s(q0 + q1)*dq0 - l2*s(q0 + q1)*dq1 - l3*s(q0 + q1 + q2)*dq0 - l3*s(q0 + q1 + q2)*dq1 - l3*s(q0 + q1 + q2)*dq2 - l4*s(q0 + q1 + q2 + q3)*dq0 - l4*s(q0 + q1 + q2 + q3)*dq1 - l4*s(q0 + q1 + q2 + q3)*dq2 - l4*s(q0 + q1 + q2 + q3)*dq3, -l2*s(q0 + q1)*dq0 - l2*s(q0 + q1)*dq1 - l3*s(q0 + q1 + q2)*dq0 - l3*s(q0 + q1 + q2)*dq1 - l3*s(q0 + q1 + q2)*dq2 - l4*s(q0 + q1 + q2 + q3)*dq0 - l4*s(q0 + q1 + q2 + q3)*dq1 - l4*s(q0 + q1 + q2 + q3)*dq2 - l4*s(q0 + q1 + q2 + q3)*dq3, -l3*s(q0 + q1 + q2)*dq0 - l3*s(q0 + q1 + q2)*dq1 - l3*s(q0 + q1 + q2)*dq2 - l4*s(q0 + q1 + q2 + q3)*dq0 - l4*s(q0 + q1 + q2 + q3)*dq1 - l4*s(q0 + q1 + q2 + q3)*dq2 - l4*s(q0 + q1 + q2 + q3)*dq3, -l4*(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3)]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8))", cache=True)
def jrx(q, i):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[-s(q0), 0., 0., 0.], [c(q0), 0., 0., 0.]])
    elif i == 1:
        return np.array([[-s(q0)*c(q1) - s(q1)*c(q0), -s(q0)*c(q1) - s(q1)*c(q0), 0., 0.], [-s(q0)*s(q1) + c(q0)*c(q1), -s(q0)*s(q1) + c(q0)*c(q1), 0., 0.]])
    elif i == 2:
        return np.array([[s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1), s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1), s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1), 0.], [-s(q0)*s(q1)*c(q2) - s(q0)*s(q2)*c(q1) - s(q1)*s(q2)*c(q0) + c(q0)*c(q1)*c(q2), -s(q0)*s(q1)*c(q2) - s(q0)*s(q2)*c(q1) - s(q1)*s(q2)*c(q0) + c(q0)*c(q1)*c(q2), -s(q0)*s(q1)*c(q2) - s(q0)*s(q2)*c(q1) - s(q1)*s(q2)*c(q0) + c(q0)*c(q1)*c(q2), 0.]])
    elif i == 3:
        return np.array([[s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)], [s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3), s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3), s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3), s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)]])
    elif i == 4:
        return np.array([[s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)], [s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3), s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3), s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3), s(q0)*s(q1)*s(q2)*s(q3) - s(q0)*s(q1)*c(q2)*c(q3) - s(q0)*s(q2)*c(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q1)*s(q2)*c(q0)*c(q3) - s(q1)*s(q3)*c(q0)*c(q2) - s(q2)*s(q3)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], f8[:, :], i8))", cache=True)
def jrx_dot(q, q_dot, i):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    dq0 = q_dot[0,0]
    dq1 = q_dot[1,0]
    dq2 = q_dot[2,0]
    dq3 = q_dot[3,0]
    if i == 0:
        return np.array([[-c(q0)*dq0, 0., 0., 0.], [-s(q0)*dq0, 0., 0., 0.]])
    elif i == 1:
        return np.array([[-(dq0 + dq1)*c(q0 + q1), -(dq0 + dq1)*c(q0 + q1), 0., 0.], [-(dq0 + dq1)*s(q0 + q1), -(dq0 + dq1)*s(q0 + q1), 0., 0.]])
    elif i == 2:
        return np.array([[-(dq0 + dq1 + dq2)*c(q0 + q1 + q2), -(dq0 + dq1 + dq2)*c(q0 + q1 + q2), -(dq0 + dq1 + dq2)*c(q0 + q1 + q2), 0], [-(dq0 + dq1 + dq2)*s(q0 + q1 + q2), -(dq0 + dq1 + dq2)*s(q0 + q1 + q2), -(dq0 + dq1 + dq2)*s(q0 + q1 + q2), 0]])
    elif i == 3:
        return np.array([[s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3, s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3, s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3, s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3], [-(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3), -(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3), -(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3), -(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3)]])
    elif i == 4:
        return np.array([[s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3, s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3, s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3, s(q0 + q1)*s(q3)*c(q2)*dq0 + s(q0 + q1)*s(q3)*c(q2)*dq1 + s(q0 + q1)*s(q3)*c(q2)*dq2 + s(q0 + q1)*s(q3)*c(q2)*dq3 + s(q1 + q2)*s(q0)*c(q3)*dq0 + s(q1 + q2)*s(q0)*c(q3)*dq1 + s(q1 + q2)*s(q0)*c(q3)*dq2 + s(q1 + q2)*s(q0)*c(q3)*dq3 + s(q1 + q3)*s(q2)*c(q0)*dq0 + s(q1 + q3)*s(q2)*c(q0)*dq1 + s(q1 + q3)*s(q2)*c(q0)*dq2 + s(q1 + q3)*s(q2)*c(q0)*dq3 - s(q0)*s(q1)*s(q2)*s(q3)*dq0 - s(q0)*s(q1)*s(q2)*s(q3)*dq1 - s(q0)*s(q1)*s(q2)*s(q3)*dq2 - s(q0)*s(q1)*s(q2)*s(q3)*dq3 - c(q0)*c(q1)*c(q2)*c(q3)*dq0 - c(q0)*c(q1)*c(q2)*c(q3)*dq1 - c(q0)*c(q1)*c(q2)*c(q3)*dq2 - c(q0)*c(q1)*c(q2)*c(q3)*dq3], [-(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3), -(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3), -(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3), -(dq0 + dq1 + dq2 + dq3)*s(q0 + q1 + q2 + q3)]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8))", cache=True)
def jry(q, i):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[-c(q0), 0., 0., 0.], [-s(q0), 0., 0., 0.]])
    elif i == 1:
        return np.array([[s(q0)*s(q1) - c(q0)*c(q1), s(q0)*s(q1) - c(q0)*c(q1), 0., 0.], [-s(q0)*c(q1) - s(q1)*c(q0), -s(q0)*c(q1) - s(q1)*c(q0), 0., 0.]])
    elif i == 2:
        return np.array([[s(q0)*s(q1)*c(q2) + s(q0)*s(q2)*c(q1) + s(q1)*s(q2)*c(q0) - c(q0)*c(q1)*c(q2), s(q0)*s(q1)*c(q2) + s(q0)*s(q2)*c(q1) + s(q1)*s(q2)*c(q0) - c(q0)*c(q1)*c(q2), s(q0)*s(q1)*c(q2) + s(q0)*s(q2)*c(q1) + s(q1)*s(q2)*c(q0) - c(q0)*c(q1)*c(q2), 0.], [s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1), s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1), s(q0)*s(q1)*s(q2) - s(q0)*c(q1)*c(q2) - s(q1)*c(q0)*c(q2) - s(q2)*c(q0)*c(q1), 0.]])
    elif i == 3:
        return np.array([[-s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3), -s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3), -s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3), -s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3)], [s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)]])
    elif i == 4:
        return np.array([[-s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3), -s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3), -s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3), -s(q0)*s(q1)*s(q2)*s(q3) + s(q0)*s(q1)*c(q2)*c(q3) + s(q0)*s(q2)*c(q1)*c(q3) + s(q0)*s(q3)*c(q1)*c(q2) + s(q1)*s(q2)*c(q0)*c(q3) + s(q1)*s(q3)*c(q0)*c(q2) + s(q2)*s(q3)*c(q0)*c(q1) - c(q0)*c(q1)*c(q2)*c(q3)], [s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2), s(q0)*s(q1)*s(q2)*c(q3) + s(q0)*s(q1)*s(q3)*c(q2) + s(q0)*s(q2)*s(q3)*c(q1) - s(q0)*c(q1)*c(q2)*c(q3) + s(q1)*s(q2)*s(q3)*c(q0) - s(q1)*c(q0)*c(q2)*c(q3) - s(q2)*c(q0)*c(q1)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8))", cache=True)
def jry_dot(q, i):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    if i == 0:
        return np.array([[-c(q0), 0., 0., 0.], [-s(q0), 0., 0., 0.]])
    elif i == 1:
        return np.array([[-c(q0 + q1), -c(q0 + q1), 0., 0.], [-s(q0 + q1), -s(q0 + q1), 0., 0.]])
    elif i == 2:
        return np.array([[-c(q0 + q1 + q2), -c(q0 + q1 + q2), -c(q0 + q1 + q2), 0.], [-s(q0 + q1 + q2), -s(q0 + q1 + q2), -s(q0 + q1 + q2), 0.]])
    elif i == 3:
        return np.array([[-c(q0 + q1 + q2 + q3), -c(q0 + q1 + q2 + q3), -c(q0 + q1 + q2 + q3), -c(q0 + q1 + q2 + q3)], [-s(q0 + q1 + q2 + q3), -s(q0 + q1 + q2 + q3), -s(q0 + q1 + q2 + q3), -s(q0 + q1 + q2 + q3)]])
    elif i == 4:
        return np.array([[-c(q0 + q1 + q2 + q3), -c(q0 + q1 + q2 + q3), -c(q0 + q1 + q2 + q3), -c(q0 + q1 + q2 + q3)], [-s(q0 + q1 + q2 + q3), -s(q0 + q1 + q2 + q3), -s(q0 + q1 + q2 + q3), -s(q0 + q1 + q2 + q3)]])
    else:
        assert False







class CPoint(mappings.Identity):
    c_dim = 4
    t_dim = 2

    q_neutral = np.array([[0, 0, 0, 0]]).T * np.pi/180  # ニュートラルの姿勢
    q_min = np.array([[-90, -90, -90, -90]]).T * np.pi/180
    q_max = np.array([[90, 90, 90, 90]]).T * np.pi/180

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

    def __init__(self, frame_num, position_num, **kwargs):
        self.l1 = kwargs.pop('l1')
        self.l2 = kwargs.pop('l2')
        self.l3 = kwargs.pop('l3')
        self.l4 = kwargs.pop('l4')
        self.o = lambda q: o(q, frame_num, self.l1, self.l2, self.l3, self.l4)
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

    def calc_all(self, q, dq):
        x = self.phi(q)
        J = self.J(q)
        x_dot = self.velocity(J, dq)
        J_dot = self.J_dot(q, dq)
        return x, x_dot, J, J_dot

    def calc_joint_position_all(self, q):
        return [
            o(q, i, self.l1, self.l2, self.l3, self.l4) for i in range(self.c_dim+1)
        ]

        
        

if __name__ == "__main__":
    pass