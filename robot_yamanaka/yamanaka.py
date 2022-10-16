import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq
from numba import njit

import mappings

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def o(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0], [0.0], [0.0]])
    elif i == 1:
        return np.array([[0.0], [0.0], [l0]])
    elif i == 2:
        return np.array([[0.0], [0.0], [l0]])
    elif i == 3:
        return np.array([[-l1*s(q[0, 0])*c(q[1, 0])], [l1*c(q[0, 0])*c(q[1, 0])], [l0 - l1*s(q[1, 0])]])
    elif i == 4:
        return np.array([[-l1*s(q[0, 0])*c(q[1, 0]) + l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [l1*c(q[0, 0])*c(q[1, 0]) - l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [l0 - l1*s(q[1, 0]) - l2*s(q[1, 0])*c(q[2, 0]) - l2*s(q[2, 0])*c(q[1, 0])]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def rx(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[1.0], [0.0], [0.0]])
    elif i == 1:
        return np.array([[-s(q[0, 0])], [c(q[0, 0])], [0.0]])
    elif i == 2:
        return np.array([[-s(q[0, 0])*c(q[1, 0])], [c(q[0, 0])*c(q[1, 0])], [-s(q[1, 0])]])
    elif i == 3:
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [-s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [-s(q[1, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[1, 0])]])
    elif i == 4:
        return np.array([[s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [-s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) + c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [-s(q[1, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[1, 0])]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def ry(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0], [1.0], [0.0]])
    elif i == 1:
        return np.array([[-c(q[0, 0])], [-s(q[0, 0])], [0.0]])
    elif i == 2:
        return np.array([[s(q[0, 0])*s(q[1, 0])], [-s(q[1, 0])*c(q[0, 0])], [-c(q[1, 0])]])
    elif i == 3:
        return np.array([[s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])], [-s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])], [s(q[1, 0])*s(q[2, 0]) - c(q[1, 0])*c(q[2, 0])]])
    elif i == 4:
        return np.array([[s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])], [-s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])], [s(q[1, 0])*s(q[2, 0]) - c(q[1, 0])*c(q[2, 0])]])
    else:
        assert False


@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def rz(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0], [0.0], [1.0]])
    elif i == 1:
        return np.array([[0.0], [0.0], [1.0]])
    elif i == 2:
        return np.array([[-c(q[0, 0])], [-s(q[0, 0])], [0.0]])
    elif i == 3:
        return np.array([[-c(q[0, 0])], [-s(q[0, 0])], [0.0]])
    elif i == 4:
        return np.array([[-c(q[0, 0])], [-s(q[0, 0])], [0.0]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def jo(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 3:
        return np.array([[-l1*c(q[0, 0])*c(q[1, 0]), l1*s(q[0, 0])*s(q[1, 0]), 0], [-l1*s(q[0, 0])*c(q[1, 0]), -l1*s(q[1, 0])*c(q[0, 0]), 0], [0, -l1*c(q[1, 0]), 0]])
    elif i == 4:
        return np.array([[-l1*c(q[0, 0])*c(q[1, 0]) + l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), l1*s(q[0, 0])*s(q[1, 0]) + l2*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + l2*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), l2*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + l2*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])], [-l1*s(q[0, 0])*c(q[1, 0]) + l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -l1*s(q[1, 0])*c(q[0, 0]) - l2*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - l2*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -l2*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - l2*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])], [0, -l1*c(q[1, 0]) + l2*s(q[1, 0])*s(q[2, 0]) - l2*c(q[1, 0])*c(q[2, 0]), l2*s(q[1, 0])*s(q[2, 0]) - l2*c(q[1, 0])*c(q[2, 0])]])
    else:
        assert False


@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def jrx(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[-c(q[0, 0]), 0, 0], [-s(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[-c(q[0, 0])*c(q[1, 0]), s(q[0, 0])*s(q[1, 0]), 0], [-s(q[0, 0])*c(q[1, 0]), -s(q[1, 0])*c(q[0, 0]), 0], [0, -c(q[1, 0]), 0]])
    elif i == 3:
        return np.array([[s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])], [0, s(q[1, 0])*s(q[2, 0]) - c(q[1, 0])*c(q[2, 0]), s(q[1, 0])*s(q[2, 0]) - c(q[1, 0])*c(q[2, 0])]])
    elif i == 4:
        return np.array([[s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])], [s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) - s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - s(q[2, 0])*c(q[0, 0])*c(q[1, 0])], [0, s(q[1, 0])*s(q[2, 0]) - c(q[1, 0])*c(q[2, 0]), s(q[1, 0])*s(q[2, 0]) - c(q[1, 0])*c(q[2, 0])]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def jry(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[s(q[1, 0])*c(q[0, 0]), s(q[0, 0])*c(q[1, 0]), 0], [s(q[0, 0])*s(q[1, 0]), -c(q[0, 0])*c(q[1, 0]), 0], [0, s(q[1, 0]), 0]])
    elif i == 3:
        return np.array([[s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [0, s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0])]])
    elif i == 4:
        return np.array([[s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [0, s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0])]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8))", cache=True)
def jrz(q, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 3:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 4:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], f8[:, :], i8, f8, f8, f8))", cache=True)
def jo_dot(q, dq, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 3:
        return np.array([[l1*s(q[0, 0])*c(q[1, 0])*dq[0, 0] + l1*s(q[1, 0])*c(q[0, 0])*dq[1, 0], l1*s(q[0, 0])*c(q[1, 0])*dq[1, 0] + l1*s(q[1, 0])*c(q[0, 0])*dq[0, 0], 0], [l1*s(q[0, 0])*s(q[1, 0])*dq[1, 0] - l1*c(q[0, 0])*c(q[1, 0])*dq[0, 0], l1*s(q[0, 0])*s(q[1, 0])*dq[0, 0] - l1*c(q[0, 0])*c(q[1, 0])*dq[1, 0], 0], [0, l1*s(q[1, 0])*dq[1, 0], 0]])
    elif i == 4:
        return np.array([[l1*s(q[0, 0])*c(q[1, 0])*dq[0, 0] + l1*s(q[1, 0])*c(q[0, 0])*dq[1, 0] - l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[0, 0] + l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[0, 0] + l2*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[1, 0] + l2*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[2, 0] + l2*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[1, 0] + l2*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[2, 0], l1*s(q[0, 0])*c(q[1, 0])*dq[1, 0] + l1*s(q[1, 0])*c(q[0, 0])*dq[0, 0] - l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[1, 0] - l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[2, 0] + l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] + l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0] + l2*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[0, 0] + l2*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[0, 0], -l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[1, 0] - l2*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[2, 0] + l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] + l2*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0] + l2*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[0, 0] + l2*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[0, 0]], [l1*s(q[0, 0])*s(q[1, 0])*dq[1, 0] - l1*c(q[0, 0])*c(q[1, 0])*dq[0, 0] + l2*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[1, 0] + l2*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[2, 0] + l2*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[1, 0] + l2*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[2, 0] + l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] - l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[0, 0], l1*s(q[0, 0])*s(q[1, 0])*dq[0, 0] - l1*c(q[0, 0])*c(q[1, 0])*dq[1, 0] + l2*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[0, 0] + l2*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[0, 0] + l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] - l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] - l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0], l2*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[0, 0] + l2*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[0, 0] + l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + l2*s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] - l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] - l2*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0]], [0, l1*s(q[1, 0])*dq[1, 0] + l2*s(q[1, 0])*c(q[2, 0])*dq[1, 0] + l2*s(q[1, 0])*c(q[2, 0])*dq[2, 0] + l2*s(q[2, 0])*c(q[1, 0])*dq[1, 0] + l2*s(q[2, 0])*c(q[1, 0])*dq[2, 0], l2*s(q[1, 0])*c(q[2, 0])*dq[1, 0] + l2*s(q[1, 0])*c(q[2, 0])*dq[2, 0] + l2*s(q[2, 0])*c(q[1, 0])*dq[1, 0] + l2*s(q[2, 0])*c(q[1, 0])*dq[2, 0]]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], f8[:, :], i8, f8, f8, f8))", cache=True)
def jrx_dot(q, dq, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[s(q[0, 0])*dq[0, 0], 0, 0], [-c(q[0, 0])*dq[0, 0], 0, 0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[s(q[0, 0])*c(q[1, 0])*dq[0, 0] + s(q[1, 0])*c(q[0, 0])*dq[1, 0], s(q[0, 0])*c(q[1, 0])*dq[1, 0] + s(q[1, 0])*c(q[0, 0])*dq[0, 0], 0], [s(q[0, 0])*s(q[1, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*dq[0, 0], s(q[0, 0])*s(q[1, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*dq[1, 0], 0], [0, s(q[1, 0])*dq[1, 0], 0]])
    elif i == 3:
        return np.array([[-s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[0, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[0, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[1, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[2, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[1, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[2, 0], -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[2, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[0, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[0, 0], -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[2, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[0, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[0, 0]], [s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[1, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[2, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[0, 0], s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[0, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0], s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[0, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0]], [0, s(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[2, 0])*c(q[1, 0])*dq[1, 0] + s(q[2, 0])*c(q[1, 0])*dq[2, 0], s(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[2, 0])*c(q[1, 0])*dq[1, 0] + s(q[2, 0])*c(q[1, 0])*dq[2, 0]]])
    elif i == 4:
        return np.array([[-s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[0, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[0, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[1, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[2, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[1, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[2, 0], -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[2, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[0, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[0, 0], -s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[1, 0] - s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*dq[2, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*dq[0, 0] + s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*dq[0, 0]], [s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[1, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[2, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[0, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[0, 0], s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[0, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0], s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*dq[0, 0] + s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*dq[0, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[1, 0] + s(q[1, 0])*s(q[2, 0])*c(q[0, 0])*dq[2, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[1, 0] - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*dq[2, 0]], [0, s(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[2, 0])*c(q[1, 0])*dq[1, 0] + s(q[2, 0])*c(q[1, 0])*dq[2, 0], s(q[1, 0])*c(q[2, 0])*dq[1, 0] + s(q[1, 0])*c(q[2, 0])*dq[2, 0] + s(q[2, 0])*c(q[1, 0])*dq[1, 0] + s(q[2, 0])*c(q[1, 0])*dq[2, 0]]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], f8[:, :], i8, f8, f8, f8))", cache=True)
def jry_dot(q, dq, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[s(q[1, 0])*c(q[0, 0]), s(q[0, 0])*c(q[1, 0]), 0], [s(q[0, 0])*s(q[1, 0]), -c(q[0, 0])*c(q[1, 0]), 0], [0, s(q[1, 0]), 0]])
    elif i == 3:
        return np.array([[s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [0, s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0])]])
    elif i == 4:
        return np.array([[s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -s(q[0, 0])*s(q[1, 0])*s(q[2, 0]) + s(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + s(q[0, 0])*s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), s(q[1, 0])*s(q[2, 0])*c(q[0, 0]) - c(q[0, 0])*c(q[1, 0])*c(q[2, 0])], [0, s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0]), s(q[1, 0])*c(q[2, 0]) + s(q[2, 0])*c(q[1, 0])]])
    else:
        assert False

@njit("(f8[:, :](f8[:, :], f8[:, :], i8, f8, f8, f8))", cache=True)
def jrz_dot(q, dq, i, l0, l1, l2):
    if i == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 1:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    elif i == 2:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 3:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    elif i == 4:
        return np.array([[s(q[0, 0]), 0, 0], [-c(q[0, 0]), 0, 0], [0.0, 0.0, 0.0]])
    else:
        assert False






class CPoint(mappings.Identity):
    c_dim = 3
    t_dim = 3

    q_neutral = np.array([[0, -90, 0]]).T * np.pi/180  # ニュートラルの姿勢
    q_min = np.array([[-90, -90-90, -90]]).T * np.pi/180
    q_max = np.array([[90, -90+90, 90]]).T * np.pi/180

    rs_in_0 = (
        (0, 0, 0),
    )  # ジョイント1によって回転する制御点

    rs_in_1 = (
        (0, 0, 0),
    )

    rs_in_2 = (
        (0, 0, 0),
    )

    rs_in_3 = (
        (0, 0, 0),
    )

    rs_in_4 = (
        (0, 0, 0),
    )

    # 追加
    RS_ALL = (
        rs_in_0, rs_in_1, rs_in_2, rs_in_3, rs_in_4,
    )


    ee_id = (4, 0)


    l0 = 1.0
    l1 = 1.0
    l2 = 1.0

    def __init__(self, frame_num, position_num, **kwargs):
        self.o = lambda q: o(q, frame_num, self.l0, self.l1, self.l2)
        self.rx = lambda q: rx(q, frame_num, self.l0, self.l1, self.l2)
        self.ry = lambda q: ry(q, frame_num, self.l0, self.l1, self.l2)
        self.jo = lambda q: jo(q, frame_num, self.l0, self.l1, self.l2)
        self.jrx = lambda q: jrx(q, frame_num, self.l0, self.l1, self.l2)
        self.jry = lambda q: jry(q, frame_num, self.l0, self.l1, self.l2)
        self.jo_dot = lambda q, q_dot: jo_dot(q, q_dot, frame_num, self.l0, self.l1, self.l2)
        self.jrx_dot = lambda q, q_dot: jrx_dot(q, q_dot, frame_num, self.l0, self.l1, self.l2)
        self.jry_dot = lambda q, q_dot: jry_dot(q, q_dot, frame_num, self.l0, self.l1, self.l2)
        self.r = self.RS_ALL[frame_num][position_num]
    
    def phi(self, q):
        return self.rx(q) * self.r[0] + self.ry(q) * self.r[1] + self.o(q)
    
    def J(self, q):
        return (self.jrx(q)*self.r[0] + self.jry(q)*self.r[1] + self.jo(q))

    def J_dot(self, q, dq):
        return self.jrx_dot(q, dq)*self.r[0] + self.jry_dot(q, dq)*self.r[1] + self.jo_dot(q, dq)

    def calc_all(self, q, dq):
        x = self.phi(q)
        J = self.J(q)
        x_dot = self.velocity(J, dq)
        J_dot = self.J_dot(q, dq)
        return x, x_dot, J, J_dot

    def calc_joint_position_all(self, q):
        return [
            o(q, i, CPoint.l0, CPoint.l1, CPoint.l2) for i in range(self.c_dim+1)
        ]





if __name__ == "__main__":
    pass