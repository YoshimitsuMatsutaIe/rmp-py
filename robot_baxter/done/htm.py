import numpy as np
from numpy.typing import NDArray
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq
from numba import njit

from numba import jit, njit


@njit("f8[:, :](f8[:, :], i8)", cache=True)
def o(q, n: int):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    
    if n == -2:
        return np.array([[-0.278], [-0.064], [1.104]])
    elif n == -1:
        return np.array([[-0.278], [-0.064], [1.37435]])
    elif n == 0:
        return np.array([[-0.278], [-0.064], [1.37435]])
    elif n == 1:
        return np.array([[0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [-0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [1.37435]])
    elif n == 2:
        return np.array([[0.36435*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [0.36435*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) - 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [1.37435 - 0.36435*s(q1)]])
    elif n == 3:
        return np.array([[-0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.36435*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2) + 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [-0.069*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.36435*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2) - 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [-0.36435*s(q1) - 0.069*c(q1)*c(q2) + 1.37435]])
    elif n == 4:
        return np.array([[-0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2) + 0.37429*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [-0.069*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2) + 0.37429*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3) - 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [-0.37429*s(q1)*c(q3) - 0.36435*s(q1) - 0.37429*s(q3)*c(q1)*c(q2) - 0.069*c(q1)*c(q2) + 1.37435]])
    elif n == 5:
        return np.array([[-0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2) + 0.01*((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + 0.37429*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.01*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4) + 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [-0.069*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2) + 0.01*((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + 0.37429*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.01*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4) - 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [0.01*(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) - 0.37429*s(q1)*c(q3) - 0.36435*s(q1) + 0.01*s(q2)*s(q4)*c(q1) - 0.37429*s(q3)*c(q1)*c(q2) - 0.069*c(q1)*c(q2) + 1.37435]])
    elif n == 6:
        return np.array([[-0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2) + 0.01*((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + 0.37429*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.01*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4) + 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [-0.069*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2) + 0.01*((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + 0.37429*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.01*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4) - 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [0.01*(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) - 0.37429*s(q1)*c(q3) - 0.36435*s(q1) + 0.01*s(q2)*s(q4)*c(q1) - 0.37429*s(q3)*c(q1)*c(q2) - 0.069*c(q1)*c(q2) + 1.37435]])
    elif n == 7:
        return np.array([[-0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2) + 0.3683*(((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) + 0.01*((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + 0.37429*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.01*(-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4) + 0.3683*((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5) + 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.278], [-0.069*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + 0.37429*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + 0.36435*(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1) + 0.069*(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2) + 0.3683*(((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) + 0.01*((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + 0.37429*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3) + 0.01*(-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4) + 0.3683*((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5) - 0.0345*sq(2)*s(q0) - 0.0345*sq(2)*c(q0) - 0.064], [0.3683*((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*s(q5) + 0.01*(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + 0.3683*(-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*c(q5) - 0.37429*s(q1)*c(q3) - 0.36435*s(q1) + 0.01*s(q2)*s(q4)*c(q1) - 0.37429*s(q3)*c(q1)*c(q2) - 0.069*c(q1)*c(q2) + 1.37435]])
    else:
        assert(False)


@njit("f8[:, :](f8[:, :], i8)", cache=True)
def rx(q, n: int):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    
    if n == -2:
        return np.array([[-0.5*sq(2)], [-0.5*sq(2)], [0.]])
    elif n == -1:
        return np.array([[-0.5*sq(2)], [-0.5*sq(2)], [0.]])
    elif n == 0:
        return np.array([[0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0)], [-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0)], [0.]])
    elif n == 1:
        return np.array([[-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)], [-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)], [-c(q1)]])
    elif n == 2:
        return np.array([[-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2)], [-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2)], [-c(q1)*c(q2)]])
    elif n == 3:
        return np.array([[-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3)], [-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3)], [s(q1)*s(q3) - c(q1)*c(q2)*c(q3)]])
    elif n == 4:
        return np.array([[((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4)], [((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4)], [(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1)]])
    elif n == 5:
        return np.array([[(((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5)], [(((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5)], [((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*c(q5) - (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*s(q5)]])
    elif n == 6:
        return np.array([[((((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*c(q6) + (((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*s(q6)], [((((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*c(q6) + (((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*s(q6)], [(((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*c(q5) - (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*s(q5))*c(q6) + (-(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*s(q4) + s(q2)*c(q1)*c(q4))*s(q6)]])
    elif n == 7:
        return np.array([[((((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*c(q6) + (((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*s(q6)], [((((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*c(q6) + (((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*s(q6)], [(((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*c(q5) - (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*s(q5))*c(q6) + (-(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*s(q4) + s(q2)*c(q1)*c(q4))*s(q6)]])
    else:
        assert(False)



@njit("f8[:, :](f8[:, :], i8)", cache=True)
def ry(q, n: int):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    q6 = q[6, 0]
    
    if n == -2:
        return np.array([[0.5*sq(2)], [-0.5*sq(2)], [0.]])
    elif n == -1:
        return np.array([[0.5*sq(2)], [-0.5*sq(2)], [0.]])
    elif n == 0:
        return np.array([[0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0)], [0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0)], [0.]])
    elif n == 1:
        return np.array([[-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)], [-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)], [s(q1)]])
    elif n == 2:
        return np.array([[(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2)], [(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2)], [s(q2)*c(q1)]])
    elif n == 3:
        return np.array([[-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3)], [-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3)], [s(q1)*c(q3) + s(q3)*c(q1)*c(q2)]])
    elif n == 4:
        return np.array([[((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4)], [((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4)], [-(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*s(q4) + s(q2)*c(q1)*c(q4)]])
    elif n == 5:
        return np.array([[-(((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) - ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5)], [-(((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) - ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5)], [-((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*s(q5) - (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*c(q5)]])
    elif n == 6:
        return np.array([[-((((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*s(q6) + (((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*c(q6)], [-((((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*s(q6) + (((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*c(q6)], [-(((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*c(q5) - (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*s(q5))*s(q6) + (-(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*s(q4) + s(q2)*c(q1)*c(q4))*c(q6)]])
    elif n == 7:
        return np.array([[-((((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*s(q6) + (((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*c(q6)], [-((((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*c(q5) - ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*s(q5))*s(q6) + (((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4))*c(q6)], [-(((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*c(q5) - (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*s(q5))*s(q6) + (-(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*s(q4) + s(q2)*c(q1)*c(q4))*c(q6)]])
    else:
        assert(False)


@njit("f8[:, :](f8[:, :], i8)", cache=True)
def rz(q, n: int):
    q0 = q[0, 0]
    q1 = q[1, 0]
    q2 = q[2, 0]
    q3 = q[3, 0]
    q4 = q[4, 0]
    q5 = q[5, 0]
    #q6 = q[6, 0]
    
    if n == -2:
        return np.array([[0.], [0.], [1.]])
    elif n == -1:
        return np.array([[0.], [0.], [1.]])
    elif n == 0:
        return np.array([[0.], [0.], [1.]])
    elif n == 1:
        return np.array([[0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0)], [0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0)], [0.]])
    elif n == 2:
        return np.array([[(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)], [(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)], [-s(q1)]])
    elif n == 3:
        return np.array([[(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2)], [(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2)], [s(q2)*c(q1)]])
    elif n == 4:
        return np.array([[(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3)], [(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3)], [-s(q1)*c(q3) - s(q3)*c(q1)*c(q2)]])
    elif n == 5:
        return np.array([[((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4)], [((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*c(q4) - (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*s(q4)], [-(s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*s(q4) + s(q2)*c(q1)*c(q4)]])
    elif n == 6:
        return np.array([[(((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) + ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5)], [(((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) + ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5)], [((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*s(q5) + (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*c(q5)]])
    elif n == 7:
        return np.array([[(((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) + ((0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) + 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5)], [(((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*s(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q2))*s(q4) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q3)*c(q1) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*c(q3))*c(q4))*s(q5) + ((-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*c(q1)*c(q3) + (-(-0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q1)*c(q2) + (0.5*sq(2)*s(q0) - 0.5*sq(2)*c(q0))*s(q2))*s(q3))*c(q5)], [((s(q1)*s(q3) - c(q1)*c(q2)*c(q3))*c(q4) + s(q2)*s(q4)*c(q1))*s(q5) + (-s(q1)*c(q3) - s(q3)*c(q1)*c(q2))*c(q5)]])
    else:
        assert(False)


if __name__ == "__main__":
    pass