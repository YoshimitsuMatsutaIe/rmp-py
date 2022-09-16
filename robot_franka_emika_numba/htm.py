import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq
from numba import jit, njit


@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8, f8, f8, f8))")
def o(q, n, d1, d3, d5, df, a4, a5, a7):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    if n == 0:
        return np.array([[0.], [0.], [d1]], dtype=np.float64)
    elif n == 1:
        return np.array([[0.], [0.], [d1]], dtype=np.float64)
    elif n == 2:
        return np.array([[d3*s(q1)*c(q0)], [d3*s(q0)*s(q1)], [d1 + d3*c(q1)]], dtype=np.float64)
    elif n == 3:
        return np.array([[-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) + d3*s(q1)*c(q0)], [a4*s(q0)*c(q1)*c(q2) + a4*s(q2)*c(q0) + d3*s(q0)*s(q1)], [-a4*s(q1)*c(q2) + d1 + d3*c(q1)]], dtype=np.float64)
    elif n == 4:
        return np.array([[-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2)], [a4*s(q0)*c(q1)*c(q2) + a4*s(q2)*c(q0) + a5*s(q0)*s(q1)*s(q3) + a5*s(q0)*c(q1)*c(q2)*c(q3) + a5*s(q2)*c(q0)*c(q3) + d3*s(q0)*s(q1) + d5*s(q0)*s(q1)*c(q3) - d5*s(q0)*s(q3)*c(q1)*c(q2) - d5*s(q2)*s(q3)*c(q0)], [-a4*s(q1)*c(q2) - a5*s(q1)*c(q2)*c(q3) + a5*s(q3)*c(q1) + d1 + d3*c(q1) + d5*s(q1)*s(q3)*c(q2) + d5*c(q1)*c(q3)]], dtype=np.float64)
    elif n == 5:
        return np.array([[-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2)], [a4*s(q0)*c(q1)*c(q2) + a4*s(q2)*c(q0) + a5*s(q0)*s(q1)*s(q3) + a5*s(q0)*c(q1)*c(q2)*c(q3) + a5*s(q2)*c(q0)*c(q3) + d3*s(q0)*s(q1) + d5*s(q0)*s(q1)*c(q3) - d5*s(q0)*s(q3)*c(q1)*c(q2) - d5*s(q2)*s(q3)*c(q0)], [-a4*s(q1)*c(q2) - a5*s(q1)*c(q2)*c(q3) + a5*s(q3)*c(q1) + d1 + d3*c(q1) + d5*s(q1)*s(q3)*c(q2) + d5*c(q1)*c(q3)]], dtype=np.float64)
    elif n == 6:
        return np.array([[-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + a7*s(q0)*s(q2)*s(q3)*s(q5) - a7*s(q0)*s(q2)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q2)*c(q5) + a7*s(q1)*s(q3)*c(q0)*c(q4)*c(q5) + a7*s(q1)*s(q5)*c(q0)*c(q3) - a7*s(q2)*s(q4)*c(q0)*c(q1)*c(q5) - a7*s(q3)*s(q5)*c(q0)*c(q1)*c(q2) + a7*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2)], [a4*s(q0)*c(q1)*c(q2) + a4*s(q2)*c(q0) + a5*s(q0)*s(q1)*s(q3) + a5*s(q0)*c(q1)*c(q2)*c(q3) + a5*s(q2)*c(q0)*c(q3) + a7*s(q0)*s(q1)*s(q3)*c(q4)*c(q5) + a7*s(q0)*s(q1)*s(q5)*c(q3) - a7*s(q0)*s(q2)*s(q4)*c(q1)*c(q5) - a7*s(q0)*s(q3)*s(q5)*c(q1)*c(q2) + a7*s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) - a7*s(q2)*s(q3)*s(q5)*c(q0) + a7*s(q2)*c(q0)*c(q3)*c(q4)*c(q5) + a7*s(q4)*c(q0)*c(q2)*c(q5) + d3*s(q0)*s(q1) + d5*s(q0)*s(q1)*c(q3) - d5*s(q0)*s(q3)*c(q1)*c(q2) - d5*s(q2)*s(q3)*c(q0)], [-a4*s(q1)*c(q2) - a5*s(q1)*c(q2)*c(q3) + a5*s(q3)*c(q1) + a7*s(q1)*s(q2)*s(q4)*c(q5) + a7*s(q1)*s(q3)*s(q5)*c(q2) - a7*s(q1)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q3)*c(q1)*c(q4)*c(q5) + a7*s(q5)*c(q1)*c(q3) + d1 + d3*c(q1) + d5*s(q1)*s(q3)*c(q2) + d5*c(q1)*c(q3)]], dtype=np.float64)
    elif n == 7:
        return np.array([[-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + a7*s(q0)*s(q2)*s(q3)*s(q5) - a7*s(q0)*s(q2)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q2)*c(q5) + a7*s(q1)*s(q3)*c(q0)*c(q4)*c(q5) + a7*s(q1)*s(q5)*c(q0)*c(q3) - a7*s(q2)*s(q4)*c(q0)*c(q1)*c(q5) - a7*s(q3)*s(q5)*c(q0)*c(q1)*c(q2) + a7*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2) - df*s(q0)*s(q2)*s(q3)*c(q5) - df*s(q0)*s(q2)*s(q5)*c(q3)*c(q4) - df*s(q0)*s(q4)*s(q5)*c(q2) + df*s(q1)*s(q3)*s(q5)*c(q0)*c(q4) - df*s(q1)*c(q0)*c(q3)*c(q5) - df*s(q2)*s(q4)*s(q5)*c(q0)*c(q1) + df*s(q3)*c(q0)*c(q1)*c(q2)*c(q5) + df*s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)], [a4*s(q0)*c(q1)*c(q2) + a4*s(q2)*c(q0) + a5*s(q0)*s(q1)*s(q3) + a5*s(q0)*c(q1)*c(q2)*c(q3) + a5*s(q2)*c(q0)*c(q3) + a7*s(q0)*s(q1)*s(q3)*c(q4)*c(q5) + a7*s(q0)*s(q1)*s(q5)*c(q3) - a7*s(q0)*s(q2)*s(q4)*c(q1)*c(q5) - a7*s(q0)*s(q3)*s(q5)*c(q1)*c(q2) + a7*s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) - a7*s(q2)*s(q3)*s(q5)*c(q0) + a7*s(q2)*c(q0)*c(q3)*c(q4)*c(q5) + a7*s(q4)*c(q0)*c(q2)*c(q5) + d3*s(q0)*s(q1) + d5*s(q0)*s(q1)*c(q3) - d5*s(q0)*s(q3)*c(q1)*c(q2) - d5*s(q2)*s(q3)*c(q0) + df*s(q0)*s(q1)*s(q3)*s(q5)*c(q4) - df*s(q0)*s(q1)*c(q3)*c(q5) - df*s(q0)*s(q2)*s(q4)*s(q5)*c(q1) + df*s(q0)*s(q3)*c(q1)*c(q2)*c(q5) + df*s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) + df*s(q2)*s(q3)*c(q0)*c(q5) + df*s(q2)*s(q5)*c(q0)*c(q3)*c(q4) + df*s(q4)*s(q5)*c(q0)*c(q2)], [-a4*s(q1)*c(q2) - a5*s(q1)*c(q2)*c(q3) + a5*s(q3)*c(q1) + a7*s(q1)*s(q2)*s(q4)*c(q5) + a7*s(q1)*s(q3)*s(q5)*c(q2) - a7*s(q1)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q3)*c(q1)*c(q4)*c(q5) + a7*s(q5)*c(q1)*c(q3) + d1 + d3*c(q1) + d5*s(q1)*s(q3)*c(q2) + d5*c(q1)*c(q3) + df*s(q1)*s(q2)*s(q4)*s(q5) - df*s(q1)*s(q3)*c(q2)*c(q5) - df*s(q1)*s(q5)*c(q2)*c(q3)*c(q4) + df*s(q3)*s(q5)*c(q1)*c(q4) - df*c(q1)*c(q3)*c(q5)]], dtype=np.float64)
    else:
        assert(False)

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8, f8, f8, f8))")
def rx(q, n, d1, d3, d5, df, a4, a5, a7):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    if n == 0:
        return np.array([[c(q0)], [s(q0)], [0.]])
    elif n == 1:
        return np.array([[c(q0)*c(q1)], [s(q0)*c(q1)], [-s(q1)]])
    elif n == 2:
        return np.array([[-s(q0)*s(q2) + c(q0)*c(q1)*c(q2)], [s(q0)*c(q1)*c(q2) + s(q2)*c(q0)], [-s(q1)*c(q2)]])
    elif n == 3:
        return np.array([[-s(q0)*s(q2)*c(q3) + s(q1)*s(q3)*c(q0) + c(q0)*c(q1)*c(q2)*c(q3)], [s(q0)*s(q1)*s(q3) + s(q0)*c(q1)*c(q2)*c(q3) + s(q2)*c(q0)*c(q3)], [-s(q1)*c(q2)*c(q3) + s(q3)*c(q1)]])
    elif n == 4:
        return np.array([[-s(q0)*s(q2)*c(q3)*c(q4) - s(q0)*s(q4)*c(q2) + s(q1)*s(q3)*c(q0)*c(q4) - s(q2)*s(q4)*c(q0)*c(q1) + c(q0)*c(q1)*c(q2)*c(q3)*c(q4)], [s(q0)*s(q1)*s(q3)*c(q4) - s(q0)*s(q2)*s(q4)*c(q1) + s(q0)*c(q1)*c(q2)*c(q3)*c(q4) + s(q2)*c(q0)*c(q3)*c(q4) + s(q4)*c(q0)*c(q2)], [s(q1)*s(q2)*s(q4) - s(q1)*c(q2)*c(q3)*c(q4) + s(q3)*c(q1)*c(q4)]])
    elif n == 5:
        return np.array([[s(q0)*s(q2)*s(q3)*s(q5) - s(q0)*s(q2)*c(q3)*c(q4)*c(q5) - s(q0)*s(q4)*c(q2)*c(q5) + s(q1)*s(q3)*c(q0)*c(q4)*c(q5) + s(q1)*s(q5)*c(q0)*c(q3) - s(q2)*s(q4)*c(q0)*c(q1)*c(q5) - s(q3)*s(q5)*c(q0)*c(q1)*c(q2) + c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)], [s(q0)*s(q1)*s(q3)*c(q4)*c(q5) + s(q0)*s(q1)*s(q5)*c(q3) - s(q0)*s(q2)*s(q4)*c(q1)*c(q5) - s(q0)*s(q3)*s(q5)*c(q1)*c(q2) + s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) - s(q2)*s(q3)*s(q5)*c(q0) + s(q2)*c(q0)*c(q3)*c(q4)*c(q5) + s(q4)*c(q0)*c(q2)*c(q5)], [s(q1)*s(q2)*s(q4)*c(q5) + s(q1)*s(q3)*s(q5)*c(q2) - s(q1)*c(q2)*c(q3)*c(q4)*c(q5) + s(q3)*c(q1)*c(q4)*c(q5) + s(q5)*c(q1)*c(q3)]])
    elif n == 6:
        return np.array([[s(q0)*s(q2)*s(q3)*s(q5)*c(q6) - s(q0)*s(q2)*s(q4)*s(q6)*c(q3) - s(q0)*s(q2)*c(q3)*c(q4)*c(q5)*c(q6) - s(q0)*s(q4)*c(q2)*c(q5)*c(q6) + s(q0)*s(q6)*c(q2)*c(q4) + s(q1)*s(q3)*s(q4)*s(q6)*c(q0) + s(q1)*s(q3)*c(q0)*c(q4)*c(q5)*c(q6) + s(q1)*s(q5)*c(q0)*c(q3)*c(q6) - s(q2)*s(q4)*c(q0)*c(q1)*c(q5)*c(q6) + s(q2)*s(q6)*c(q0)*c(q1)*c(q4) - s(q3)*s(q5)*c(q0)*c(q1)*c(q2)*c(q6) + s(q4)*s(q6)*c(q0)*c(q1)*c(q2)*c(q3) + c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)*c(q6)], [s(q0)*s(q1)*s(q3)*s(q4)*s(q6) + s(q0)*s(q1)*s(q3)*c(q4)*c(q5)*c(q6) + s(q0)*s(q1)*s(q5)*c(q3)*c(q6) - s(q0)*s(q2)*s(q4)*c(q1)*c(q5)*c(q6) + s(q0)*s(q2)*s(q6)*c(q1)*c(q4) - s(q0)*s(q3)*s(q5)*c(q1)*c(q2)*c(q6) + s(q0)*s(q4)*s(q6)*c(q1)*c(q2)*c(q3) + s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)*c(q6) - s(q2)*s(q3)*s(q5)*c(q0)*c(q6) + s(q2)*s(q4)*s(q6)*c(q0)*c(q3) + s(q2)*c(q0)*c(q3)*c(q4)*c(q5)*c(q6) + s(q4)*c(q0)*c(q2)*c(q5)*c(q6) - s(q6)*c(q0)*c(q2)*c(q4)], [s(q1)*s(q2)*s(q4)*c(q5)*c(q6) - s(q1)*s(q2)*s(q6)*c(q4) + s(q1)*s(q3)*s(q5)*c(q2)*c(q6) - s(q1)*s(q4)*s(q6)*c(q2)*c(q3) - s(q1)*c(q2)*c(q3)*c(q4)*c(q5)*c(q6) + s(q3)*s(q4)*s(q6)*c(q1) + s(q3)*c(q1)*c(q4)*c(q5)*c(q6) + s(q5)*c(q1)*c(q3)*c(q6)]])
    elif n == 7:
        return np.array([[s(q0)*s(q2)*s(q3)*s(q5)*c(q6) - s(q0)*s(q2)*s(q4)*s(q6)*c(q3) - s(q0)*s(q2)*c(q3)*c(q4)*c(q5)*c(q6) - s(q0)*s(q4)*c(q2)*c(q5)*c(q6) + s(q0)*s(q6)*c(q2)*c(q4) + s(q1)*s(q3)*s(q4)*s(q6)*c(q0) + s(q1)*s(q3)*c(q0)*c(q4)*c(q5)*c(q6) + s(q1)*s(q5)*c(q0)*c(q3)*c(q6) - s(q2)*s(q4)*c(q0)*c(q1)*c(q5)*c(q6) + s(q2)*s(q6)*c(q0)*c(q1)*c(q4) - s(q3)*s(q5)*c(q0)*c(q1)*c(q2)*c(q6) + s(q4)*s(q6)*c(q0)*c(q1)*c(q2)*c(q3) + c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)*c(q6)], [s(q0)*s(q1)*s(q3)*s(q4)*s(q6) + s(q0)*s(q1)*s(q3)*c(q4)*c(q5)*c(q6) + s(q0)*s(q1)*s(q5)*c(q3)*c(q6) - s(q0)*s(q2)*s(q4)*c(q1)*c(q5)*c(q6) + s(q0)*s(q2)*s(q6)*c(q1)*c(q4) - s(q0)*s(q3)*s(q5)*c(q1)*c(q2)*c(q6) + s(q0)*s(q4)*s(q6)*c(q1)*c(q2)*c(q3) + s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)*c(q6) - s(q2)*s(q3)*s(q5)*c(q0)*c(q6) + s(q2)*s(q4)*s(q6)*c(q0)*c(q3) + s(q2)*c(q0)*c(q3)*c(q4)*c(q5)*c(q6) + s(q4)*c(q0)*c(q2)*c(q5)*c(q6) - s(q6)*c(q0)*c(q2)*c(q4)], [s(q1)*s(q2)*s(q4)*c(q5)*c(q6) - s(q1)*s(q2)*s(q6)*c(q4) + s(q1)*s(q3)*s(q5)*c(q2)*c(q6) - s(q1)*s(q4)*s(q6)*c(q2)*c(q3) - s(q1)*c(q2)*c(q3)*c(q4)*c(q5)*c(q6) + s(q3)*s(q4)*s(q6)*c(q1) + s(q3)*c(q1)*c(q4)*c(q5)*c(q6) + s(q5)*c(q1)*c(q3)*c(q6)]])
    else:
        assert(False)
    
@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8, f8, f8, f8))")
def ry(q, n, d1, d3, d5, df, a4, a5, a7):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    if n == 0:
        return np.array([[-s(q0)], [c(q0)], [0.]])
    elif n == 1:
        return np.array([[-s(q1)*c(q0)], [-s(q0)*s(q1)], [-c(q1)]])
    elif n == 2:
        return np.array([[-s(q0)*c(q2) - s(q2)*c(q0)*c(q1)], [-s(q0)*s(q2)*c(q1) + c(q0)*c(q2)], [s(q1)*s(q2)]])
    elif n == 3:
        return np.array([[s(q0)*s(q2)*s(q3) + s(q1)*c(q0)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)], [s(q0)*s(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q2)*s(q3)*c(q0)], [s(q1)*s(q3)*c(q2) + c(q1)*c(q3)]])
    elif n == 4:
        return np.array([[s(q0)*s(q2)*s(q4)*c(q3) - s(q0)*c(q2)*c(q4) - s(q1)*s(q3)*s(q4)*c(q0) - s(q2)*c(q0)*c(q1)*c(q4) - s(q4)*c(q0)*c(q1)*c(q2)*c(q3)], [-s(q0)*s(q1)*s(q3)*s(q4) - s(q0)*s(q2)*c(q1)*c(q4) - s(q0)*s(q4)*c(q1)*c(q2)*c(q3) - s(q2)*s(q4)*c(q0)*c(q3) + c(q0)*c(q2)*c(q4)], [s(q1)*s(q2)*c(q4) + s(q1)*s(q4)*c(q2)*c(q3) - s(q3)*s(q4)*c(q1)]])
    elif n == 5:
        return np.array([[s(q0)*s(q2)*s(q3)*c(q5) + s(q0)*s(q2)*s(q5)*c(q3)*c(q4) + s(q0)*s(q4)*s(q5)*c(q2) - s(q1)*s(q3)*s(q5)*c(q0)*c(q4) + s(q1)*c(q0)*c(q3)*c(q5) + s(q2)*s(q4)*s(q5)*c(q0)*c(q1) - s(q3)*c(q0)*c(q1)*c(q2)*c(q5) - s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)], [-s(q0)*s(q1)*s(q3)*s(q5)*c(q4) + s(q0)*s(q1)*c(q3)*c(q5) + s(q0)*s(q2)*s(q4)*s(q5)*c(q1) - s(q0)*s(q3)*c(q1)*c(q2)*c(q5) - s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) - s(q2)*s(q3)*c(q0)*c(q5) - s(q2)*s(q5)*c(q0)*c(q3)*c(q4) - s(q4)*s(q5)*c(q0)*c(q2)], [-s(q1)*s(q2)*s(q4)*s(q5) + s(q1)*s(q3)*c(q2)*c(q5) + s(q1)*s(q5)*c(q2)*c(q3)*c(q4) - s(q3)*s(q5)*c(q1)*c(q4) + c(q1)*c(q3)*c(q5)]])
    elif n == 6:
        return np.array([[-s(q0)*s(q2)*s(q3)*s(q5)*s(q6) - s(q0)*s(q2)*s(q4)*c(q3)*c(q6) + s(q0)*s(q2)*s(q6)*c(q3)*c(q4)*c(q5) + s(q0)*s(q4)*s(q6)*c(q2)*c(q5) + s(q0)*c(q2)*c(q4)*c(q6) + s(q1)*s(q3)*s(q4)*c(q0)*c(q6) - s(q1)*s(q3)*s(q6)*c(q0)*c(q4)*c(q5) - s(q1)*s(q5)*s(q6)*c(q0)*c(q3) + s(q2)*s(q4)*s(q6)*c(q0)*c(q1)*c(q5) + s(q2)*c(q0)*c(q1)*c(q4)*c(q6) + s(q3)*s(q5)*s(q6)*c(q0)*c(q1)*c(q2) + s(q4)*c(q0)*c(q1)*c(q2)*c(q3)*c(q6) - s(q6)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)], [s(q0)*s(q1)*s(q3)*s(q4)*c(q6) - s(q0)*s(q1)*s(q3)*s(q6)*c(q4)*c(q5) - s(q0)*s(q1)*s(q5)*s(q6)*c(q3) + s(q0)*s(q2)*s(q4)*s(q6)*c(q1)*c(q5) + s(q0)*s(q2)*c(q1)*c(q4)*c(q6) + s(q0)*s(q3)*s(q5)*s(q6)*c(q1)*c(q2) + s(q0)*s(q4)*c(q1)*c(q2)*c(q3)*c(q6) - s(q0)*s(q6)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + s(q2)*s(q3)*s(q5)*s(q6)*c(q0) + s(q2)*s(q4)*c(q0)*c(q3)*c(q6) - s(q2)*s(q6)*c(q0)*c(q3)*c(q4)*c(q5) - s(q4)*s(q6)*c(q0)*c(q2)*c(q5) - c(q0)*c(q2)*c(q4)*c(q6)], [-s(q1)*s(q2)*s(q4)*s(q6)*c(q5) - s(q1)*s(q2)*c(q4)*c(q6) - s(q1)*s(q3)*s(q5)*s(q6)*c(q2) - s(q1)*s(q4)*c(q2)*c(q3)*c(q6) + s(q1)*s(q6)*c(q2)*c(q3)*c(q4)*c(q5) + s(q3)*s(q4)*c(q1)*c(q6) - s(q3)*s(q6)*c(q1)*c(q4)*c(q5) - s(q5)*s(q6)*c(q1)*c(q3)]])
    elif n == 7:
        return np.array([[-s(q0)*s(q2)*s(q3)*s(q5)*s(q6) - s(q0)*s(q2)*s(q4)*c(q3)*c(q6) + s(q0)*s(q2)*s(q6)*c(q3)*c(q4)*c(q5) + s(q0)*s(q4)*s(q6)*c(q2)*c(q5) + s(q0)*c(q2)*c(q4)*c(q6) + s(q1)*s(q3)*s(q4)*c(q0)*c(q6) - s(q1)*s(q3)*s(q6)*c(q0)*c(q4)*c(q5) - s(q1)*s(q5)*s(q6)*c(q0)*c(q3) + s(q2)*s(q4)*s(q6)*c(q0)*c(q1)*c(q5) + s(q2)*c(q0)*c(q1)*c(q4)*c(q6) + s(q3)*s(q5)*s(q6)*c(q0)*c(q1)*c(q2) + s(q4)*c(q0)*c(q1)*c(q2)*c(q3)*c(q6) - s(q6)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5)], [s(q0)*s(q1)*s(q3)*s(q4)*c(q6) - s(q0)*s(q1)*s(q3)*s(q6)*c(q4)*c(q5) - s(q0)*s(q1)*s(q5)*s(q6)*c(q3) + s(q0)*s(q2)*s(q4)*s(q6)*c(q1)*c(q5) + s(q0)*s(q2)*c(q1)*c(q4)*c(q6) + s(q0)*s(q3)*s(q5)*s(q6)*c(q1)*c(q2) + s(q0)*s(q4)*c(q1)*c(q2)*c(q3)*c(q6) - s(q0)*s(q6)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + s(q2)*s(q3)*s(q5)*s(q6)*c(q0) + s(q2)*s(q4)*c(q0)*c(q3)*c(q6) - s(q2)*s(q6)*c(q0)*c(q3)*c(q4)*c(q5) - s(q4)*s(q6)*c(q0)*c(q2)*c(q5) - c(q0)*c(q2)*c(q4)*c(q6)], [-s(q1)*s(q2)*s(q4)*s(q6)*c(q5) - s(q1)*s(q2)*c(q4)*c(q6) - s(q1)*s(q3)*s(q5)*s(q6)*c(q2) - s(q1)*s(q4)*c(q2)*c(q3)*c(q6) + s(q1)*s(q6)*c(q2)*c(q3)*c(q4)*c(q5) + s(q3)*s(q4)*c(q1)*c(q6) - s(q3)*s(q6)*c(q1)*c(q4)*c(q5) - s(q5)*s(q6)*c(q1)*c(q3)]])
    else:
        assert(False)

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8, f8, f8, f8))")
def rz(q, n, d1, d3, d5, df, a4, a5, a7):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    if n == 0:
        return np.array([[0.], [0.], [1.]])
    elif n == 1:
        return np.array([[-s(q0)], [c(q0)], [0.]])
    elif n == 2:
        return np.array([[s(q1)*c(q0)], [s(q0)*s(q1)], [c(q1)]])
    elif n == 3:
        return np.array([[s(q0)*c(q2) + s(q2)*c(q0)*c(q1)], [s(q0)*s(q2)*c(q1) - c(q0)*c(q2)], [-s(q1)*s(q2)]])
    elif n == 4:
        return np.array([[s(q0)*s(q2)*s(q3) + s(q1)*c(q0)*c(q3) - s(q3)*c(q0)*c(q1)*c(q2)], [s(q0)*s(q1)*c(q3) - s(q0)*s(q3)*c(q1)*c(q2) - s(q2)*s(q3)*c(q0)], [s(q1)*s(q3)*c(q2) + c(q1)*c(q3)]])
    elif n == 5:
        return np.array([[-s(q0)*s(q2)*s(q4)*c(q3) + s(q0)*c(q2)*c(q4) + s(q1)*s(q3)*s(q4)*c(q0) + s(q2)*c(q0)*c(q1)*c(q4) + s(q4)*c(q0)*c(q1)*c(q2)*c(q3)], [s(q0)*s(q1)*s(q3)*s(q4) + s(q0)*s(q2)*c(q1)*c(q4) + s(q0)*s(q4)*c(q1)*c(q2)*c(q3) + s(q2)*s(q4)*c(q0)*c(q3) - c(q0)*c(q2)*c(q4)], [-s(q1)*s(q2)*c(q4) - s(q1)*s(q4)*c(q2)*c(q3) + s(q3)*s(q4)*c(q1)]])
    elif n == 6:
        return np.array([[-s(q0)*s(q2)*s(q3)*c(q5) - s(q0)*s(q2)*s(q5)*c(q3)*c(q4) - s(q0)*s(q4)*s(q5)*c(q2) + s(q1)*s(q3)*s(q5)*c(q0)*c(q4) - s(q1)*c(q0)*c(q3)*c(q5) - s(q2)*s(q4)*s(q5)*c(q0)*c(q1) + s(q3)*c(q0)*c(q1)*c(q2)*c(q5) + s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)], [s(q0)*s(q1)*s(q3)*s(q5)*c(q4) - s(q0)*s(q1)*c(q3)*c(q5) - s(q0)*s(q2)*s(q4)*s(q5)*c(q1) + s(q0)*s(q3)*c(q1)*c(q2)*c(q5) + s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) + s(q2)*s(q3)*c(q0)*c(q5) + s(q2)*s(q5)*c(q0)*c(q3)*c(q4) + s(q4)*s(q5)*c(q0)*c(q2)], [s(q1)*s(q2)*s(q4)*s(q5) - s(q1)*s(q3)*c(q2)*c(q5) - s(q1)*s(q5)*c(q2)*c(q3)*c(q4) + s(q3)*s(q5)*c(q1)*c(q4) - c(q1)*c(q3)*c(q5)]])
    elif n == 7:
        return np.array([[-s(q0)*s(q2)*s(q3)*c(q5) - s(q0)*s(q2)*s(q5)*c(q3)*c(q4) - s(q0)*s(q4)*s(q5)*c(q2) + s(q1)*s(q3)*s(q5)*c(q0)*c(q4) - s(q1)*c(q0)*c(q3)*c(q5) - s(q2)*s(q4)*s(q5)*c(q0)*c(q1) + s(q3)*c(q0)*c(q1)*c(q2)*c(q5) + s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)], [s(q0)*s(q1)*s(q3)*s(q5)*c(q4) - s(q0)*s(q1)*c(q3)*c(q5) - s(q0)*s(q2)*s(q4)*s(q5)*c(q1) + s(q0)*s(q3)*c(q1)*c(q2)*c(q5) + s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) + s(q2)*s(q3)*c(q0)*c(q5) + s(q2)*s(q5)*c(q0)*c(q3)*c(q4) + s(q4)*s(q5)*c(q0)*c(q2)], [s(q1)*s(q2)*s(q4)*s(q5) - s(q1)*s(q3)*c(q2)*c(q5) - s(q1)*s(q5)*c(q2)*c(q3)*c(q4) + s(q3)*s(q5)*c(q1)*c(q4) - c(q1)*c(q3)*c(q5)]])
    else:
        assert(False)


if __name__ == "__main__":
    pass