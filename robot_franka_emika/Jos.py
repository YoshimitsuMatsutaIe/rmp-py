import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq
from numba import njit

@njit("(f8[:, :](f8[:, :], i8, f8, f8, f8, f8, f8, f8, f8))", cache=True)
def jo(q, n, d1, d3, d5, df, a4, a5, a7):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]

    if n == 0:
        return np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.]])
    elif n == 1:
        return np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.]])
    elif n == 2:
        return np.array([[-d3*s(q0)*s(q1), d3*c(q0)*c(q1), 0., 0., 0., 0., 0.], [d3*s(q1)*c(q0), d3*s(q0)*c(q1), 0., 0., 0., 0., 0.], [0., -d3*s(q1), 0., 0., 0., 0., 0.]])
    elif n == 3:
        return np.array([[-a4*s(q0)*c(q1)*c(q2) - a4*s(q2)*c(q0) - d3*s(q0)*s(q1), -a4*s(q1)*c(q0)*c(q2) + d3*c(q0)*c(q1), -a4*s(q0)*c(q2) - a4*s(q2)*c(q0)*c(q1), 0., 0., 0., 0.], [-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) + d3*s(q1)*c(q0), -a4*s(q0)*s(q1)*c(q2) + d3*s(q0)*c(q1), -a4*s(q0)*s(q2)*c(q1) + a4*c(q0)*c(q2), 0., 0., 0., 0.], [0., -a4*c(q1)*c(q2) - d3*s(q1), a4*s(q1)*s(q2), 0., 0., 0., 0.]])
    elif n == 4:
        return np.array([[-a4*s(q0)*c(q1)*c(q2) - a4*s(q2)*c(q0) - a5*s(q0)*s(q1)*s(q3) - a5*s(q0)*c(q1)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q3) - d3*s(q0)*s(q1) - d5*s(q0)*s(q1)*c(q3) + d5*s(q0)*s(q3)*c(q1)*c(q2) + d5*s(q2)*s(q3)*c(q0), -a4*s(q1)*c(q0)*c(q2) - a5*s(q1)*c(q0)*c(q2)*c(q3) + a5*s(q3)*c(q0)*c(q1) + d3*c(q0)*c(q1) + d5*s(q1)*s(q3)*c(q0)*c(q2) + d5*c(q0)*c(q1)*c(q3), -a4*s(q0)*c(q2) - a4*s(q2)*c(q0)*c(q1) - a5*s(q0)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q1)*c(q3) + d5*s(q0)*s(q3)*c(q2) + d5*s(q2)*s(q3)*c(q0)*c(q1), a5*s(q0)*s(q2)*s(q3) + a5*s(q1)*c(q0)*c(q3) - a5*s(q3)*c(q0)*c(q1)*c(q2) + d5*s(q0)*s(q2)*c(q3) - d5*s(q1)*s(q3)*c(q0) - d5*c(q0)*c(q1)*c(q2)*c(q3), 0., 0., 0.], [-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2), -a4*s(q0)*s(q1)*c(q2) - a5*s(q0)*s(q1)*c(q2)*c(q3) + a5*s(q0)*s(q3)*c(q1) + d3*s(q0)*c(q1) + d5*s(q0)*s(q1)*s(q3)*c(q2) + d5*s(q0)*c(q1)*c(q3), -a4*s(q0)*s(q2)*c(q1) + a4*c(q0)*c(q2) - a5*s(q0)*s(q2)*c(q1)*c(q3) + a5*c(q0)*c(q2)*c(q3) + d5*s(q0)*s(q2)*s(q3)*c(q1) - d5*s(q3)*c(q0)*c(q2), a5*s(q0)*s(q1)*c(q3) - a5*s(q0)*s(q3)*c(q1)*c(q2) - a5*s(q2)*s(q3)*c(q0) - d5*s(q0)*s(q1)*s(q3) - d5*s(q0)*c(q1)*c(q2)*c(q3) - d5*s(q2)*c(q0)*c(q3), 0., 0., 0.], [0., -a4*c(q1)*c(q2) - a5*s(q1)*s(q3) - a5*c(q1)*c(q2)*c(q3) - d3*s(q1) - d5*s(q1)*c(q3) + d5*s(q3)*c(q1)*c(q2), a4*s(q1)*s(q2) + a5*s(q1)*s(q2)*c(q3) - d5*s(q1)*s(q2)*s(q3), a5*s(q1)*s(q3)*c(q2) + a5*c(q1)*c(q3) + d5*s(q1)*c(q2)*c(q3) - d5*s(q3)*c(q1), 0., 0., 0.]])
    elif n == 5:
        return np.array([[-a4*s(q0)*c(q1)*c(q2) - a4*s(q2)*c(q0) - a5*s(q0)*s(q1)*s(q3) - a5*s(q0)*c(q1)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q3) - d3*s(q0)*s(q1) - d5*s(q0)*s(q1)*c(q3) + d5*s(q0)*s(q3)*c(q1)*c(q2) + d5*s(q2)*s(q3)*c(q0), -a4*s(q1)*c(q0)*c(q2) - a5*s(q1)*c(q0)*c(q2)*c(q3) + a5*s(q3)*c(q0)*c(q1) + d3*c(q0)*c(q1) + d5*s(q1)*s(q3)*c(q0)*c(q2) + d5*c(q0)*c(q1)*c(q3), -a4*s(q0)*c(q2) - a4*s(q2)*c(q0)*c(q1) - a5*s(q0)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q1)*c(q3) + d5*s(q0)*s(q3)*c(q2) + d5*s(q2)*s(q3)*c(q0)*c(q1), a5*s(q0)*s(q2)*s(q3) + a5*s(q1)*c(q0)*c(q3) - a5*s(q3)*c(q0)*c(q1)*c(q2) + d5*s(q0)*s(q2)*c(q3) - d5*s(q1)*s(q3)*c(q0) - d5*c(q0)*c(q1)*c(q2)*c(q3), 0., 0., 0.], [-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2), -a4*s(q0)*s(q1)*c(q2) - a5*s(q0)*s(q1)*c(q2)*c(q3) + a5*s(q0)*s(q3)*c(q1) + d3*s(q0)*c(q1) + d5*s(q0)*s(q1)*s(q3)*c(q2) + d5*s(q0)*c(q1)*c(q3), -a4*s(q0)*s(q2)*c(q1) + a4*c(q0)*c(q2) - a5*s(q0)*s(q2)*c(q1)*c(q3) + a5*c(q0)*c(q2)*c(q3) + d5*s(q0)*s(q2)*s(q3)*c(q1) - d5*s(q3)*c(q0)*c(q2), a5*s(q0)*s(q1)*c(q3) - a5*s(q0)*s(q3)*c(q1)*c(q2) - a5*s(q2)*s(q3)*c(q0) - d5*s(q0)*s(q1)*s(q3) - d5*s(q0)*c(q1)*c(q2)*c(q3) - d5*s(q2)*c(q0)*c(q3), 0., 0., 0.], [0., -a4*c(q1)*c(q2) - a5*s(q1)*s(q3) - a5*c(q1)*c(q2)*c(q3) - d3*s(q1) - d5*s(q1)*c(q3) + d5*s(q3)*c(q1)*c(q2), a4*s(q1)*s(q2) + a5*s(q1)*s(q2)*c(q3) - d5*s(q1)*s(q2)*s(q3), a5*s(q1)*s(q3)*c(q2) + a5*c(q1)*c(q3) + d5*s(q1)*c(q2)*c(q3) - d5*s(q3)*c(q1), 0., 0., 0.]])
    elif n == 6:
        return np.array([[-a4*s(q0)*c(q1)*c(q2) - a4*s(q2)*c(q0) - a5*s(q0)*s(q1)*s(q3) - a5*s(q0)*c(q1)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q3) - a7*s(q0)*s(q1)*s(q3)*c(q4)*c(q5) - a7*s(q0)*s(q1)*s(q5)*c(q3) + a7*s(q0)*s(q2)*s(q4)*c(q1)*c(q5) + a7*s(q0)*s(q3)*s(q5)*c(q1)*c(q2) - a7*s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q2)*s(q3)*s(q5)*c(q0) - a7*s(q2)*c(q0)*c(q3)*c(q4)*c(q5) - a7*s(q4)*c(q0)*c(q2)*c(q5) - d3*s(q0)*s(q1) - d5*s(q0)*s(q1)*c(q3) + d5*s(q0)*s(q3)*c(q1)*c(q2) + d5*s(q2)*s(q3)*c(q0), -a4*s(q1)*c(q0)*c(q2) - a5*s(q1)*c(q0)*c(q2)*c(q3) + a5*s(q3)*c(q0)*c(q1) + a7*s(q1)*s(q2)*s(q4)*c(q0)*c(q5) + a7*s(q1)*s(q3)*s(q5)*c(q0)*c(q2) - a7*s(q1)*c(q0)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q3)*c(q0)*c(q1)*c(q4)*c(q5) + a7*s(q5)*c(q0)*c(q1)*c(q3) + d3*c(q0)*c(q1) + d5*s(q1)*s(q3)*c(q0)*c(q2) + d5*c(q0)*c(q1)*c(q3), -a4*s(q0)*c(q2) - a4*s(q2)*c(q0)*c(q1) - a5*s(q0)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q1)*c(q3) + a7*s(q0)*s(q2)*s(q4)*c(q5) + a7*s(q0)*s(q3)*s(q5)*c(q2) - a7*s(q0)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q2)*s(q3)*s(q5)*c(q0)*c(q1) - a7*s(q2)*c(q0)*c(q1)*c(q3)*c(q4)*c(q5) - a7*s(q4)*c(q0)*c(q1)*c(q2)*c(q5) + d5*s(q0)*s(q3)*c(q2) + d5*s(q2)*s(q3)*c(q0)*c(q1), a5*s(q0)*s(q2)*s(q3) + a5*s(q1)*c(q0)*c(q3) - a5*s(q3)*c(q0)*c(q1)*c(q2) + a7*s(q0)*s(q2)*s(q3)*c(q4)*c(q5) + a7*s(q0)*s(q2)*s(q5)*c(q3) - a7*s(q1)*s(q3)*s(q5)*c(q0) + a7*s(q1)*c(q0)*c(q3)*c(q4)*c(q5) - a7*s(q3)*c(q0)*c(q1)*c(q2)*c(q4)*c(q5) - a7*s(q5)*c(q0)*c(q1)*c(q2)*c(q3) + d5*s(q0)*s(q2)*c(q3) - d5*s(q1)*s(q3)*c(q0) - d5*c(q0)*c(q1)*c(q2)*c(q3), a7*s(q0)*s(q2)*s(q4)*c(q3)*c(q5) - a7*s(q0)*c(q2)*c(q4)*c(q5) - a7*s(q1)*s(q3)*s(q4)*c(q0)*c(q5) - a7*s(q2)*c(q0)*c(q1)*c(q4)*c(q5) - a7*s(q4)*c(q0)*c(q1)*c(q2)*c(q3)*c(q5), a7*s(q0)*s(q2)*s(q3)*c(q5) + a7*s(q0)*s(q2)*s(q5)*c(q3)*c(q4) + a7*s(q0)*s(q4)*s(q5)*c(q2) - a7*s(q1)*s(q3)*s(q5)*c(q0)*c(q4) + a7*s(q1)*c(q0)*c(q3)*c(q5) + a7*s(q2)*s(q4)*s(q5)*c(q0)*c(q1) - a7*s(q3)*c(q0)*c(q1)*c(q2)*c(q5) - a7*s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4), 0.], [-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + a7*s(q0)*s(q2)*s(q3)*s(q5) - a7*s(q0)*s(q2)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q2)*c(q5) + a7*s(q1)*s(q3)*c(q0)*c(q4)*c(q5) + a7*s(q1)*s(q5)*c(q0)*c(q3) - a7*s(q2)*s(q4)*c(q0)*c(q1)*c(q5) - a7*s(q3)*s(q5)*c(q0)*c(q1)*c(q2) + a7*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2), -a4*s(q0)*s(q1)*c(q2) - a5*s(q0)*s(q1)*c(q2)*c(q3) + a5*s(q0)*s(q3)*c(q1) + a7*s(q0)*s(q1)*s(q2)*s(q4)*c(q5) + a7*s(q0)*s(q1)*s(q3)*s(q5)*c(q2) - a7*s(q0)*s(q1)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q0)*s(q3)*c(q1)*c(q4)*c(q5) + a7*s(q0)*s(q5)*c(q1)*c(q3) + d3*s(q0)*c(q1) + d5*s(q0)*s(q1)*s(q3)*c(q2) + d5*s(q0)*c(q1)*c(q3), -a4*s(q0)*s(q2)*c(q1) + a4*c(q0)*c(q2) - a5*s(q0)*s(q2)*c(q1)*c(q3) + a5*c(q0)*c(q2)*c(q3) + a7*s(q0)*s(q2)*s(q3)*s(q5)*c(q1) - a7*s(q0)*s(q2)*c(q1)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q1)*c(q2)*c(q5) - a7*s(q2)*s(q4)*c(q0)*c(q5) - a7*s(q3)*s(q5)*c(q0)*c(q2) + a7*c(q0)*c(q2)*c(q3)*c(q4)*c(q5) + d5*s(q0)*s(q2)*s(q3)*c(q1) - d5*s(q3)*c(q0)*c(q2), a5*s(q0)*s(q1)*c(q3) - a5*s(q0)*s(q3)*c(q1)*c(q2) - a5*s(q2)*s(q3)*c(q0) - a7*s(q0)*s(q1)*s(q3)*s(q5) + a7*s(q0)*s(q1)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q3)*c(q1)*c(q2)*c(q4)*c(q5) - a7*s(q0)*s(q5)*c(q1)*c(q2)*c(q3) - a7*s(q2)*s(q3)*c(q0)*c(q4)*c(q5) - a7*s(q2)*s(q5)*c(q0)*c(q3) - d5*s(q0)*s(q1)*s(q3) - d5*s(q0)*c(q1)*c(q2)*c(q3) - d5*s(q2)*c(q0)*c(q3), -a7*s(q0)*s(q1)*s(q3)*s(q4)*c(q5) - a7*s(q0)*s(q2)*c(q1)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q1)*c(q2)*c(q3)*c(q5) - a7*s(q2)*s(q4)*c(q0)*c(q3)*c(q5) + a7*c(q0)*c(q2)*c(q4)*c(q5), -a7*s(q0)*s(q1)*s(q3)*s(q5)*c(q4) + a7*s(q0)*s(q1)*c(q3)*c(q5) + a7*s(q0)*s(q2)*s(q4)*s(q5)*c(q1) - a7*s(q0)*s(q3)*c(q1)*c(q2)*c(q5) - a7*s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) - a7*s(q2)*s(q3)*c(q0)*c(q5) - a7*s(q2)*s(q5)*c(q0)*c(q3)*c(q4) - a7*s(q4)*s(q5)*c(q0)*c(q2), 0.], [0., -a4*c(q1)*c(q2) - a5*s(q1)*s(q3) - a5*c(q1)*c(q2)*c(q3) - a7*s(q1)*s(q3)*c(q4)*c(q5) - a7*s(q1)*s(q5)*c(q3) + a7*s(q2)*s(q4)*c(q1)*c(q5) + a7*s(q3)*s(q5)*c(q1)*c(q2) - a7*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) - d3*s(q1) - d5*s(q1)*c(q3) + d5*s(q3)*c(q1)*c(q2), a4*s(q1)*s(q2) + a5*s(q1)*s(q2)*c(q3) - a7*s(q1)*s(q2)*s(q3)*s(q5) + a7*s(q1)*s(q2)*c(q3)*c(q4)*c(q5) + a7*s(q1)*s(q4)*c(q2)*c(q5) - d5*s(q1)*s(q2)*s(q3), a5*s(q1)*s(q3)*c(q2) + a5*c(q1)*c(q3) + a7*s(q1)*s(q3)*c(q2)*c(q4)*c(q5) + a7*s(q1)*s(q5)*c(q2)*c(q3) - a7*s(q3)*s(q5)*c(q1) + a7*c(q1)*c(q3)*c(q4)*c(q5) + d5*s(q1)*c(q2)*c(q3) - d5*s(q3)*c(q1), a7*s(q1)*s(q2)*c(q4)*c(q5) + a7*s(q1)*s(q4)*c(q2)*c(q3)*c(q5) - a7*s(q3)*s(q4)*c(q1)*c(q5), -a7*s(q1)*s(q2)*s(q4)*s(q5) + a7*s(q1)*s(q3)*c(q2)*c(q5) + a7*s(q1)*s(q5)*c(q2)*c(q3)*c(q4) - a7*s(q3)*s(q5)*c(q1)*c(q4) + a7*c(q1)*c(q3)*c(q5), 0.]])
    elif n == 7:
        return np.array([[-a4*s(q0)*c(q1)*c(q2) - a4*s(q2)*c(q0) - a5*s(q0)*s(q1)*s(q3) - a5*s(q0)*c(q1)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q3) - a7*s(q0)*s(q1)*s(q3)*c(q4)*c(q5) - a7*s(q0)*s(q1)*s(q5)*c(q3) + a7*s(q0)*s(q2)*s(q4)*c(q1)*c(q5) + a7*s(q0)*s(q3)*s(q5)*c(q1)*c(q2) - a7*s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q2)*s(q3)*s(q5)*c(q0) - a7*s(q2)*c(q0)*c(q3)*c(q4)*c(q5) - a7*s(q4)*c(q0)*c(q2)*c(q5) - d3*s(q0)*s(q1) - d5*s(q0)*s(q1)*c(q3) + d5*s(q0)*s(q3)*c(q1)*c(q2) + d5*s(q2)*s(q3)*c(q0) - df*s(q0)*s(q1)*s(q3)*s(q5)*c(q4) + df*s(q0)*s(q1)*c(q3)*c(q5) + df*s(q0)*s(q2)*s(q4)*s(q5)*c(q1) - df*s(q0)*s(q3)*c(q1)*c(q2)*c(q5) - df*s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) - df*s(q2)*s(q3)*c(q0)*c(q5) - df*s(q2)*s(q5)*c(q0)*c(q3)*c(q4) - df*s(q4)*s(q5)*c(q0)*c(q2), -a4*s(q1)*c(q0)*c(q2) - a5*s(q1)*c(q0)*c(q2)*c(q3) + a5*s(q3)*c(q0)*c(q1) + a7*s(q1)*s(q2)*s(q4)*c(q0)*c(q5) + a7*s(q1)*s(q3)*s(q5)*c(q0)*c(q2) - a7*s(q1)*c(q0)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q3)*c(q0)*c(q1)*c(q4)*c(q5) + a7*s(q5)*c(q0)*c(q1)*c(q3) + d3*c(q0)*c(q1) + d5*s(q1)*s(q3)*c(q0)*c(q2) + d5*c(q0)*c(q1)*c(q3) + df*s(q1)*s(q2)*s(q4)*s(q5)*c(q0) - df*s(q1)*s(q3)*c(q0)*c(q2)*c(q5) - df*s(q1)*s(q5)*c(q0)*c(q2)*c(q3)*c(q4) + df*s(q3)*s(q5)*c(q0)*c(q1)*c(q4) - df*c(q0)*c(q1)*c(q3)*c(q5), -a4*s(q0)*c(q2) - a4*s(q2)*c(q0)*c(q1) - a5*s(q0)*c(q2)*c(q3) - a5*s(q2)*c(q0)*c(q1)*c(q3) + a7*s(q0)*s(q2)*s(q4)*c(q5) + a7*s(q0)*s(q3)*s(q5)*c(q2) - a7*s(q0)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q2)*s(q3)*s(q5)*c(q0)*c(q1) - a7*s(q2)*c(q0)*c(q1)*c(q3)*c(q4)*c(q5) - a7*s(q4)*c(q0)*c(q1)*c(q2)*c(q5) + d5*s(q0)*s(q3)*c(q2) + d5*s(q2)*s(q3)*c(q0)*c(q1) + df*s(q0)*s(q2)*s(q4)*s(q5) - df*s(q0)*s(q3)*c(q2)*c(q5) - df*s(q0)*s(q5)*c(q2)*c(q3)*c(q4) - df*s(q2)*s(q3)*c(q0)*c(q1)*c(q5) - df*s(q2)*s(q5)*c(q0)*c(q1)*c(q3)*c(q4) - df*s(q4)*s(q5)*c(q0)*c(q1)*c(q2), a5*s(q0)*s(q2)*s(q3) + a5*s(q1)*c(q0)*c(q3) - a5*s(q3)*c(q0)*c(q1)*c(q2) + a7*s(q0)*s(q2)*s(q3)*c(q4)*c(q5) + a7*s(q0)*s(q2)*s(q5)*c(q3) - a7*s(q1)*s(q3)*s(q5)*c(q0) + a7*s(q1)*c(q0)*c(q3)*c(q4)*c(q5) - a7*s(q3)*c(q0)*c(q1)*c(q2)*c(q4)*c(q5) - a7*s(q5)*c(q0)*c(q1)*c(q2)*c(q3) + d5*s(q0)*s(q2)*c(q3) - d5*s(q1)*s(q3)*c(q0) - d5*c(q0)*c(q1)*c(q2)*c(q3) + df*s(q0)*s(q2)*s(q3)*s(q5)*c(q4) - df*s(q0)*s(q2)*c(q3)*c(q5) + df*s(q1)*s(q3)*c(q0)*c(q5) + df*s(q1)*s(q5)*c(q0)*c(q3)*c(q4) - df*s(q3)*s(q5)*c(q0)*c(q1)*c(q2)*c(q4) + df*c(q0)*c(q1)*c(q2)*c(q3)*c(q5), a7*s(q0)*s(q2)*s(q4)*c(q3)*c(q5) - a7*s(q0)*c(q2)*c(q4)*c(q5) - a7*s(q1)*s(q3)*s(q4)*c(q0)*c(q5) - a7*s(q2)*c(q0)*c(q1)*c(q4)*c(q5) - a7*s(q4)*c(q0)*c(q1)*c(q2)*c(q3)*c(q5) + df*s(q0)*s(q2)*s(q4)*s(q5)*c(q3) - df*s(q0)*s(q5)*c(q2)*c(q4) - df*s(q1)*s(q3)*s(q4)*s(q5)*c(q0) - df*s(q2)*s(q5)*c(q0)*c(q1)*c(q4) - df*s(q4)*s(q5)*c(q0)*c(q1)*c(q2)*c(q3), a7*s(q0)*s(q2)*s(q3)*c(q5) + a7*s(q0)*s(q2)*s(q5)*c(q3)*c(q4) + a7*s(q0)*s(q4)*s(q5)*c(q2) - a7*s(q1)*s(q3)*s(q5)*c(q0)*c(q4) + a7*s(q1)*c(q0)*c(q3)*c(q5) + a7*s(q2)*s(q4)*s(q5)*c(q0)*c(q1) - a7*s(q3)*c(q0)*c(q1)*c(q2)*c(q5) - a7*s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4) + df*s(q0)*s(q2)*s(q3)*s(q5) - df*s(q0)*s(q2)*c(q3)*c(q4)*c(q5) - df*s(q0)*s(q4)*c(q2)*c(q5) + df*s(q1)*s(q3)*c(q0)*c(q4)*c(q5) + df*s(q1)*s(q5)*c(q0)*c(q3) - df*s(q2)*s(q4)*c(q0)*c(q1)*c(q5) - df*s(q3)*s(q5)*c(q0)*c(q1)*c(q2) + df*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5), 0.], [-a4*s(q0)*s(q2) + a4*c(q0)*c(q1)*c(q2) - a5*s(q0)*s(q2)*c(q3) + a5*s(q1)*s(q3)*c(q0) + a5*c(q0)*c(q1)*c(q2)*c(q3) + a7*s(q0)*s(q2)*s(q3)*s(q5) - a7*s(q0)*s(q2)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q2)*c(q5) + a7*s(q1)*s(q3)*c(q0)*c(q4)*c(q5) + a7*s(q1)*s(q5)*c(q0)*c(q3) - a7*s(q2)*s(q4)*c(q0)*c(q1)*c(q5) - a7*s(q3)*s(q5)*c(q0)*c(q1)*c(q2) + a7*c(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) + d3*s(q1)*c(q0) + d5*s(q0)*s(q2)*s(q3) + d5*s(q1)*c(q0)*c(q3) - d5*s(q3)*c(q0)*c(q1)*c(q2) - df*s(q0)*s(q2)*s(q3)*c(q5) - df*s(q0)*s(q2)*s(q5)*c(q3)*c(q4) - df*s(q0)*s(q4)*s(q5)*c(q2) + df*s(q1)*s(q3)*s(q5)*c(q0)*c(q4) - df*s(q1)*c(q0)*c(q3)*c(q5) - df*s(q2)*s(q4)*s(q5)*c(q0)*c(q1) + df*s(q3)*c(q0)*c(q1)*c(q2)*c(q5) + df*s(q5)*c(q0)*c(q1)*c(q2)*c(q3)*c(q4), -a4*s(q0)*s(q1)*c(q2) - a5*s(q0)*s(q1)*c(q2)*c(q3) + a5*s(q0)*s(q3)*c(q1) + a7*s(q0)*s(q1)*s(q2)*s(q4)*c(q5) + a7*s(q0)*s(q1)*s(q3)*s(q5)*c(q2) - a7*s(q0)*s(q1)*c(q2)*c(q3)*c(q4)*c(q5) + a7*s(q0)*s(q3)*c(q1)*c(q4)*c(q5) + a7*s(q0)*s(q5)*c(q1)*c(q3) + d3*s(q0)*c(q1) + d5*s(q0)*s(q1)*s(q3)*c(q2) + d5*s(q0)*c(q1)*c(q3) + df*s(q0)*s(q1)*s(q2)*s(q4)*s(q5) - df*s(q0)*s(q1)*s(q3)*c(q2)*c(q5) - df*s(q0)*s(q1)*s(q5)*c(q2)*c(q3)*c(q4) + df*s(q0)*s(q3)*s(q5)*c(q1)*c(q4) - df*s(q0)*c(q1)*c(q3)*c(q5), -a4*s(q0)*s(q2)*c(q1) + a4*c(q0)*c(q2) - a5*s(q0)*s(q2)*c(q1)*c(q3) + a5*c(q0)*c(q2)*c(q3) + a7*s(q0)*s(q2)*s(q3)*s(q5)*c(q1) - a7*s(q0)*s(q2)*c(q1)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q1)*c(q2)*c(q5) - a7*s(q2)*s(q4)*c(q0)*c(q5) - a7*s(q3)*s(q5)*c(q0)*c(q2) + a7*c(q0)*c(q2)*c(q3)*c(q4)*c(q5) + d5*s(q0)*s(q2)*s(q3)*c(q1) - d5*s(q3)*c(q0)*c(q2) - df*s(q0)*s(q2)*s(q3)*c(q1)*c(q5) - df*s(q0)*s(q2)*s(q5)*c(q1)*c(q3)*c(q4) - df*s(q0)*s(q4)*s(q5)*c(q1)*c(q2) - df*s(q2)*s(q4)*s(q5)*c(q0) + df*s(q3)*c(q0)*c(q2)*c(q5) + df*s(q5)*c(q0)*c(q2)*c(q3)*c(q4), a5*s(q0)*s(q1)*c(q3) - a5*s(q0)*s(q3)*c(q1)*c(q2) - a5*s(q2)*s(q3)*c(q0) - a7*s(q0)*s(q1)*s(q3)*s(q5) + a7*s(q0)*s(q1)*c(q3)*c(q4)*c(q5) - a7*s(q0)*s(q3)*c(q1)*c(q2)*c(q4)*c(q5) - a7*s(q0)*s(q5)*c(q1)*c(q2)*c(q3) - a7*s(q2)*s(q3)*c(q0)*c(q4)*c(q5) - a7*s(q2)*s(q5)*c(q0)*c(q3) - d5*s(q0)*s(q1)*s(q3) - d5*s(q0)*c(q1)*c(q2)*c(q3) - d5*s(q2)*c(q0)*c(q3) + df*s(q0)*s(q1)*s(q3)*c(q5) + df*s(q0)*s(q1)*s(q5)*c(q3)*c(q4) - df*s(q0)*s(q3)*s(q5)*c(q1)*c(q2)*c(q4) + df*s(q0)*c(q1)*c(q2)*c(q3)*c(q5) - df*s(q2)*s(q3)*s(q5)*c(q0)*c(q4) + df*s(q2)*c(q0)*c(q3)*c(q5), -a7*s(q0)*s(q1)*s(q3)*s(q4)*c(q5) - a7*s(q0)*s(q2)*c(q1)*c(q4)*c(q5) - a7*s(q0)*s(q4)*c(q1)*c(q2)*c(q3)*c(q5) - a7*s(q2)*s(q4)*c(q0)*c(q3)*c(q5) + a7*c(q0)*c(q2)*c(q4)*c(q5) - df*s(q0)*s(q1)*s(q3)*s(q4)*s(q5) - df*s(q0)*s(q2)*s(q5)*c(q1)*c(q4) - df*s(q0)*s(q4)*s(q5)*c(q1)*c(q2)*c(q3) - df*s(q2)*s(q4)*s(q5)*c(q0)*c(q3) + df*s(q5)*c(q0)*c(q2)*c(q4), -a7*s(q0)*s(q1)*s(q3)*s(q5)*c(q4) + a7*s(q0)*s(q1)*c(q3)*c(q5) + a7*s(q0)*s(q2)*s(q4)*s(q5)*c(q1) - a7*s(q0)*s(q3)*c(q1)*c(q2)*c(q5) - a7*s(q0)*s(q5)*c(q1)*c(q2)*c(q3)*c(q4) - a7*s(q2)*s(q3)*c(q0)*c(q5) - a7*s(q2)*s(q5)*c(q0)*c(q3)*c(q4) - a7*s(q4)*s(q5)*c(q0)*c(q2) + df*s(q0)*s(q1)*s(q3)*c(q4)*c(q5) + df*s(q0)*s(q1)*s(q5)*c(q3) - df*s(q0)*s(q2)*s(q4)*c(q1)*c(q5) - df*s(q0)*s(q3)*s(q5)*c(q1)*c(q2) + df*s(q0)*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) - df*s(q2)*s(q3)*s(q5)*c(q0) + df*s(q2)*c(q0)*c(q3)*c(q4)*c(q5) + df*s(q4)*c(q0)*c(q2)*c(q5), 0.], [0., -a4*c(q1)*c(q2) - a5*s(q1)*s(q3) - a5*c(q1)*c(q2)*c(q3) - a7*s(q1)*s(q3)*c(q4)*c(q5) - a7*s(q1)*s(q5)*c(q3) + a7*s(q2)*s(q4)*c(q1)*c(q5) + a7*s(q3)*s(q5)*c(q1)*c(q2) - a7*c(q1)*c(q2)*c(q3)*c(q4)*c(q5) - d3*s(q1) - d5*s(q1)*c(q3) + d5*s(q3)*c(q1)*c(q2) - df*s(q1)*s(q3)*s(q5)*c(q4) + df*s(q1)*c(q3)*c(q5) + df*s(q2)*s(q4)*s(q5)*c(q1) - df*s(q3)*c(q1)*c(q2)*c(q5) - df*s(q5)*c(q1)*c(q2)*c(q3)*c(q4), a4*s(q1)*s(q2) + a5*s(q1)*s(q2)*c(q3) - a7*s(q1)*s(q2)*s(q3)*s(q5) + a7*s(q1)*s(q2)*c(q3)*c(q4)*c(q5) + a7*s(q1)*s(q4)*c(q2)*c(q5) - d5*s(q1)*s(q2)*s(q3) + df*s(q1)*s(q2)*s(q3)*c(q5) + df*s(q1)*s(q2)*s(q5)*c(q3)*c(q4) + df*s(q1)*s(q4)*s(q5)*c(q2), a5*s(q1)*s(q3)*c(q2) + a5*c(q1)*c(q3) + a7*s(q1)*s(q3)*c(q2)*c(q4)*c(q5) + a7*s(q1)*s(q5)*c(q2)*c(q3) - a7*s(q3)*s(q5)*c(q1) + a7*c(q1)*c(q3)*c(q4)*c(q5) + d5*s(q1)*c(q2)*c(q3) - d5*s(q3)*c(q1) + df*s(q1)*s(q3)*s(q5)*c(q2)*c(q4) - df*s(q1)*c(q2)*c(q3)*c(q5) + df*s(q3)*c(q1)*c(q5) + df*s(q5)*c(q1)*c(q3)*c(q4), a7*s(q1)*s(q2)*c(q4)*c(q5) + a7*s(q1)*s(q4)*c(q2)*c(q3)*c(q5) - a7*s(q3)*s(q4)*c(q1)*c(q5) + df*s(q1)*s(q2)*s(q5)*c(q4) + df*s(q1)*s(q4)*s(q5)*c(q2)*c(q3) - df*s(q3)*s(q4)*s(q5)*c(q1), -a7*s(q1)*s(q2)*s(q4)*s(q5) + a7*s(q1)*s(q3)*c(q2)*c(q5) + a7*s(q1)*s(q5)*c(q2)*c(q3)*c(q4) - a7*s(q3)*s(q5)*c(q1)*c(q4) + a7*c(q1)*c(q3)*c(q5) + df*s(q1)*s(q2)*s(q4)*c(q5) + df*s(q1)*s(q3)*s(q5)*c(q2) - df*s(q1)*c(q2)*c(q3)*c(q4)*c(q5) + df*s(q3)*c(q1)*c(q4)*c(q5) + df*s(q5)*c(q1)*c(q3), 0.]])
    else:
        assert(False)




if __name__ == "__main__":
    pass