import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq
from numba import njit

@njit("(f8[:, :](f8[:, :], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8))", cache=True)
def M(q, l0, l1, l2, r9, r1, r2, Ix1, Ix2, Ix3, Iy1, Iy2, Iy3, Iz1, Iz2, Iz3, m1, m2, m3, g):
    return np.array([
        [Iy2*s(q[1, 0])**2 + Iy3*s(q[1, 0] + q[2, 0])**2 + Iz1 + Iz2*c(q[1, 0])**2 + Iz3*c(q[1, 0] + q[2, 0])**2 + m2*r1**2*c(q[1, 0])**2 + m3*(l1*c(q[1, 0]) + r2*c(q[1, 0] + q[2, 0]))**2, 0., 0.],
        [0., Ix2 + Ix3 + l1**2*m3 + 2*l1*m3*r2*c(q[2, 0]) + m2*r1**2 + m3*r2**2, Ix3 + l1*m3*r2 + m3*r2**2 + c(q[2, 0])],
        [0., Ix3 + l1*m3*r2*c(q[2, 0]) + m3*r2**2, Ix3 + m3*r2**2]
    ])

@njit("(f8[:, :](f8[:, :], f8[:, :], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8))", cache=True)
def C(q, dq, l0, l1, l2, r9, r1, r2, Ix1, Ix2, Ix3, Iy1, Iy2, Iy3, Iz1, Iz2, Iz3, m1, m2, m3, g):
    return np.array([
        [(-m3*r2*(l1*c(q[1, 0]) + r2 + c(q[1, 0] + q[2, 0]))*s(q[1, 0] + q[2, 0]) + (Iy3 - Iz3)*s(q[1, 0] + q[2, 0])*c(q[1, 0] + q[2, 0]))*dq[2, 0] + (-m3*(l1*s(q[1, 0]) + r2*s(q[1, 0] + q[2, 0]))*(l1*c(q[1, 0]) + r2*c(q[1, 0] + q[2, 0])) + (Iy3 - Iz3)*s(q[1, 0] + q[2, 0])*c(q[1, 0] + q[2, 0]) + (Iy2 - Iz2 - m2*r1**2)*s(q[1, 0])*c(q[1, 0]))*dq[1, 0], (-m3*(l1*s(q[1, 0]) + r2*s(q[1, 0] + q[2, 0]))*(l1*c(q[1, 0]) + r2*c(q[1, 0] + q[2, 0])) + (Iy3 - Iz3)*s(q[1, 0] + q[2, 0])*c(q[1, 0] + q[2, 0]) + (Iy2 - Iz2 - m2*r1**2)*s(q[1, 0])*c(q[1, 0]))*dq[0, 0], (-m3*r2*(l1*c(q[1, 0]) + r2 + c(q[1, 0] + q[2, 0]))*s(q[1, 0] + q[2, 0]) + (Iy3 - Iz3)*s(q[1, 0] + q[2, 0])*c(q[1, 0] + q[2, 0]))*dq[0, 0]],
        [(m3*(l1*c(q[1, 0]) + r2*c(q[1, 0] + q[2, 0]))*(l1*s(q[1, 0]) + r2 + s(q[1, 0] + q[2, 0])) - (Iy3 - Iz3)*s(q[1, 0] + q[2, 0])*c(q[1, 0] + q[2, 0]) + (-Iy2 + Iz2 + m2*r1**2)*s(q[1, 0])*c(q[1, 0]))*dq[0, 0], -l1*m3*r2*s(q[2, 0])*dq[2, 0], -l1*m3*r2*s(q[2, 0])*dq[1, 0] - l1*m3*r2*s(q[2, 0])*dq[2, 0]],
        [(m3*r2*(l1*c(q[1, 0]) + r2*c(q[1, 0] + q[2, 0]))*s(q[1, 0] + q[2, 0]) + (-Iy3 + Iz3)*s(q[1, 0] + q[2, 0])*c(q[1, 0] + q[2, 0]))*dq[0, 0], l1*m3*r2*s(q[2, 0])*dq[1, 0], 0.]
    ])

@njit("(f8[:, :](f8[:, :], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8))", cache=True)
def G(q, l0, l1, l2, r9, r1, r2, Ix1, Ix2, Ix3, Iy1, Iy2, Iy3, Iz1, Iz2, Iz3, m1, m2, m3, g):
    return np.array([
        [0.],
        [-g*m3*r2*c(q[1, 0] + q[2, 0]) + (-g*l1*m3 - g*m2*r1)*c(q[1, 0])],
        [-g*m3*r2*c(q[1, 0] + q[2, 0])]
    ])
