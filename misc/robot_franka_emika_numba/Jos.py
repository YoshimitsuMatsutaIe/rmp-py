import numpy as np
from math import cos as c
from math import sin as s
from math import tan as ta
from math import sqrt as sq
d1 = 0.333
d3 = 0.316
d5 = 0.384
df = 0.107
a4 = 0.0825
a5 = -0.0825
a7 = 0.088
def jo_0(q):
    return np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
def jo_1(q):
    return np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
def jo_2(q):
    return np.array([[-d3*s(q[0, 0])*s(q[1, 0]), d3*c(q[0, 0])*c(q[1, 0]), 0, 0, 0, 0, 0], [d3*s(q[1, 0])*c(q[0, 0]), d3*s(q[0, 0])*c(q[1, 0]), 0, 0, 0, 0, 0], [0, -d3*s(q[1, 0]), 0, 0, 0, 0, 0]])
def jo_3(q):
    return np.array([[-a4*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0]) - d3*s(q[0, 0])*s(q[1, 0]), -a4*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) + d3*c(q[0, 0])*c(q[1, 0]), -a4*s(q[0, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]), 0, 0, 0, 0], [-a4*s(q[0, 0])*s(q[2, 0]) + a4*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + d3*s(q[1, 0])*c(q[0, 0]), -a4*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) + d3*s(q[0, 0])*c(q[1, 0]), -a4*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + a4*c(q[0, 0])*c(q[2, 0]), 0, 0, 0, 0], [0, -a4*c(q[1, 0])*c(q[2, 0]) - d3*s(q[1, 0]), a4*s(q[1, 0])*s(q[2, 0]), 0, 0, 0, 0]])
def jo_4(q):
    return np.array([[-a4*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0]) - a5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - a5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - d3*s(q[0, 0])*s(q[1, 0]) - d5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]), -a4*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - a5*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + d3*c(q[0, 0])*c(q[1, 0]) + d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + d5*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]), -a4*s(q[0, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]) - a5*s(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]), a5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + a5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + d5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) - d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) - d5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), 0, 0, 0], [-a4*s(q[0, 0])*s(q[2, 0]) + a4*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) + a5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) + a5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + d3*s(q[1, 0])*c(q[0, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + d5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -a4*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0]) + d3*s(q[0, 0])*c(q[1, 0]) + d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[0, 0])*c(q[1, 0])*c(q[3, 0]), -a4*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + a4*c(q[0, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + a5*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]), a5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) - a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - d5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]), 0, 0, 0], [0, -a4*c(q[1, 0])*c(q[2, 0]) - a5*s(q[1, 0])*s(q[3, 0]) - a5*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d3*s(q[1, 0]) - d5*s(q[1, 0])*c(q[3, 0]) + d5*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]), a4*s(q[1, 0])*s(q[2, 0]) + a5*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - d5*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]), a5*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + a5*c(q[1, 0])*c(q[3, 0]) + d5*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[1, 0]), 0, 0, 0]])
def jo_5(q):
    return np.array([[-a4*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0]) - a5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - a5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - d3*s(q[0, 0])*s(q[1, 0]) - d5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]), -a4*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - a5*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + d3*c(q[0, 0])*c(q[1, 0]) + d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + d5*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]), -a4*s(q[0, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]) - a5*s(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]), a5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + a5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + d5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) - d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) - d5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), 0, 0, 0], [-a4*s(q[0, 0])*s(q[2, 0]) + a4*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) + a5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) + a5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + d3*s(q[1, 0])*c(q[0, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + d5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -a4*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0]) + d3*s(q[0, 0])*c(q[1, 0]) + d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[0, 0])*c(q[1, 0])*c(q[3, 0]), -a4*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + a4*c(q[0, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + a5*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]), a5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) - a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - d5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]), 0, 0, 0], [0, -a4*c(q[1, 0])*c(q[2, 0]) - a5*s(q[1, 0])*s(q[3, 0]) - a5*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d3*s(q[1, 0]) - d5*s(q[1, 0])*c(q[3, 0]) + d5*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]), a4*s(q[1, 0])*s(q[2, 0]) + a5*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - d5*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]), a5*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + a5*c(q[1, 0])*c(q[3, 0]) + d5*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[1, 0]), 0, 0, 0]])
def jo_6(q):
    return np.array([[-a4*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0]) - a5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - a5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[1, 0])*s(q[5, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0]) - a7*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0]) - a7*s(q[2, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[4, 0])*c(q[0, 0])*c(q[2, 0])*c(q[5, 0]) - d3*s(q[0, 0])*s(q[1, 0]) - d5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]), -a4*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - a5*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + a7*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]) - a7*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + d3*c(q[0, 0])*c(q[1, 0]) + d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + d5*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]), -a4*s(q[0, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]) - a5*s(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[2, 0]) - a7*s(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0]) - a7*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]), a5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + a5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[3, 0]) - a7*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0]) + a7*s(q[1, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) - d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) - d5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[3, 0])*c(q[5, 0]) - a7*s(q[0, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[1, 0])*s(q[3, 0])*s(q[4, 0])*c(q[0, 0])*c(q[5, 0]) - a7*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]), a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[3, 0])*c(q[4, 0]) + a7*s(q[0, 0])*s(q[4, 0])*s(q[5, 0])*c(q[2, 0]) - a7*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[4, 0]) + a7*s(q[1, 0])*c(q[0, 0])*c(q[3, 0])*c(q[5, 0]) + a7*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0]) - a7*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - a7*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]), 0], [-a4*s(q[0, 0])*s(q[2, 0]) + a4*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) + a5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) + a5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0]) - a7*s(q[0, 0])*s(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[4, 0])*c(q[2, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0]) - a7*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[5, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + a7*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + d3*s(q[1, 0])*c(q[0, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + d5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), -a4*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0]) + a7*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[2, 0]) - a7*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[3, 0]) + d3*s(q[0, 0])*c(q[1, 0]) + d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[0, 0])*c(q[1, 0])*c(q[3, 0]), -a4*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + a4*c(q[0, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + a5*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0]) - a7*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[4, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[5, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]) + a7*c(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]), a5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) - a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0]) + a7*s(q[0, 0])*s(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a7*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0]) - d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - d5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]), -a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[4, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[3, 0])*c(q[5, 0]) + a7*c(q[0, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]), -a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[4, 0]) + a7*s(q[0, 0])*s(q[1, 0])*c(q[3, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0]) - a7*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) - a7*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0]) - a7*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]), 0], [0, -a4*c(q[1, 0])*c(q[2, 0]) - a5*s(q[1, 0])*s(q[3, 0]) - a5*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a7*s(q[1, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[1, 0])*s(q[5, 0])*c(q[3, 0]) + a7*s(q[2, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]) + a7*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0]) - a7*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - d3*s(q[1, 0]) - d5*s(q[1, 0])*c(q[3, 0]) + d5*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]), a4*s(q[1, 0])*s(q[2, 0]) + a5*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - a7*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0]) + a7*s(q[1, 0])*s(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[4, 0])*c(q[2, 0])*c(q[5, 0]) - d5*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]), a5*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + a5*c(q[1, 0])*c(q[3, 0]) + a7*s(q[1, 0])*s(q[3, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[1, 0]) + a7*c(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + d5*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[1, 0]), a7*s(q[1, 0])*s(q[2, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[4, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) - a7*s(q[3, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]), -a7*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0]) + a7*s(q[1, 0])*s(q[3, 0])*c(q[2, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[4, 0]) + a7*c(q[1, 0])*c(q[3, 0])*c(q[5, 0]), 0]])
def jo_ee(q):
    return np.array([[-a4*s(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0]) - a5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - a5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) - a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[1, 0])*s(q[5, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0]) - a7*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0]) - a7*s(q[2, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[4, 0])*c(q[0, 0])*c(q[2, 0])*c(q[5, 0]) - d3*s(q[0, 0])*s(q[1, 0]) - d5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - df*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[4, 0]) + df*s(q[0, 0])*s(q[1, 0])*c(q[3, 0])*c(q[5, 0]) + df*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0]) - df*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[5, 0]) - df*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]), -a4*s(q[1, 0])*c(q[0, 0])*c(q[2, 0]) - a5*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + a7*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]) - a7*s(q[1, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + d3*c(q[0, 0])*c(q[1, 0]) + d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) + d5*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + df*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0]) - df*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[2, 0])*c(q[5, 0]) - df*s(q[1, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) + df*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[4, 0]) - df*c(q[0, 0])*c(q[1, 0])*c(q[3, 0])*c(q[5, 0]), -a4*s(q[0, 0])*c(q[2, 0]) - a4*s(q[2, 0])*c(q[0, 0])*c(q[1, 0]) - a5*s(q[0, 0])*c(q[2, 0])*c(q[3, 0]) - a5*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[2, 0]) - a7*s(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0]) - a7*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) + d5*s(q[0, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0]) + df*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0]) - df*s(q[0, 0])*s(q[3, 0])*c(q[2, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[5, 0]) - df*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]), a5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + a5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - a5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[3, 0]) - a7*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0]) + a7*s(q[1, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + d5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) - d5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) - d5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + df*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[4, 0]) - df*s(q[0, 0])*s(q[2, 0])*c(q[3, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[4, 0]) + df*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]), a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[3, 0])*c(q[5, 0]) - a7*s(q[0, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[1, 0])*s(q[3, 0])*s(q[4, 0])*c(q[0, 0])*c(q[5, 0]) - a7*s(q[2, 0])*c(q[0, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) + df*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[3, 0]) - df*s(q[0, 0])*s(q[5, 0])*c(q[2, 0])*c(q[4, 0]) - df*s(q[1, 0])*s(q[3, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0]) - df*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[4, 0]) - df*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]), a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[3, 0])*c(q[4, 0]) + a7*s(q[0, 0])*s(q[4, 0])*s(q[5, 0])*c(q[2, 0]) - a7*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[4, 0]) + a7*s(q[1, 0])*c(q[0, 0])*c(q[3, 0])*c(q[5, 0]) + a7*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0]) - a7*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - a7*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) + df*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0]) - df*s(q[0, 0])*s(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[4, 0])*c(q[2, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[4, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0]) - df*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[5, 0]) - df*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + df*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]), 0], [-a4*s(q[0, 0])*s(q[2, 0]) + a4*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[3, 0]) + a5*s(q[1, 0])*s(q[3, 0])*c(q[0, 0]) + a5*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0]) - a7*s(q[0, 0])*s(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[4, 0])*c(q[2, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[3, 0])*c(q[0, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0]) - a7*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[1, 0])*c(q[5, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) + a7*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + d3*s(q[1, 0])*c(q[0, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0]) + d5*s(q[1, 0])*c(q[0, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0]) - df*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[0, 0])*s(q[4, 0])*s(q[5, 0])*c(q[2, 0]) + df*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[4, 0]) - df*s(q[1, 0])*c(q[0, 0])*c(q[3, 0])*c(q[5, 0]) - df*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[1, 0]) + df*s(q[3, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) + df*s(q[5, 0])*c(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]), -a4*s(q[0, 0])*s(q[1, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) + a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0]) + a7*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[2, 0]) - a7*s(q[0, 0])*s(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[3, 0]) + d3*s(q[0, 0])*c(q[1, 0]) + d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + d5*s(q[0, 0])*c(q[1, 0])*c(q[3, 0]) + df*s(q[0, 0])*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0]) - df*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[2, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[1, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) + df*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[4, 0]) - df*s(q[0, 0])*c(q[1, 0])*c(q[3, 0])*c(q[5, 0]), -a4*s(q[0, 0])*s(q[2, 0])*c(q[1, 0]) + a4*c(q[0, 0])*c(q[2, 0]) - a5*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0]) + a5*c(q[0, 0])*c(q[2, 0])*c(q[3, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0]) - a7*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[4, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[5, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]) + a7*c(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + d5*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0]) - d5*s(q[3, 0])*c(q[0, 0])*c(q[2, 0]) - df*s(q[0, 0])*s(q[2, 0])*s(q[3, 0])*c(q[1, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[0, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0]) - df*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0]) + df*s(q[3, 0])*c(q[0, 0])*c(q[2, 0])*c(q[5, 0]) + df*s(q[5, 0])*c(q[0, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]), a5*s(q[0, 0])*s(q[1, 0])*c(q[3, 0]) - a5*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - a5*s(q[2, 0])*s(q[3, 0])*c(q[0, 0]) - a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0]) + a7*s(q[0, 0])*s(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a7*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0]) - d5*s(q[0, 0])*s(q[1, 0])*s(q[3, 0]) - d5*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[2, 0])*c(q[0, 0])*c(q[3, 0]) + df*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[5, 0]) + df*s(q[0, 0])*s(q[1, 0])*s(q[5, 0])*c(q[3, 0])*c(q[4, 0]) - df*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[4, 0]) + df*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) - df*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0])*c(q[4, 0]) + df*s(q[2, 0])*c(q[0, 0])*c(q[3, 0])*c(q[5, 0]), -a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[2, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[4, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[4, 0])*c(q[0, 0])*c(q[3, 0])*c(q[5, 0]) + a7*c(q[0, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[4, 0])*s(q[5, 0]) - df*s(q[0, 0])*s(q[2, 0])*s(q[5, 0])*c(q[1, 0])*c(q[4, 0]) - df*s(q[0, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - df*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0]) + df*s(q[5, 0])*c(q[0, 0])*c(q[2, 0])*c(q[4, 0]), -a7*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[4, 0]) + a7*s(q[0, 0])*s(q[1, 0])*c(q[3, 0])*c(q[5, 0]) + a7*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0]) - a7*s(q[0, 0])*s(q[3, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - a7*s(q[0, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) - a7*s(q[2, 0])*s(q[3, 0])*c(q[0, 0])*c(q[5, 0]) - a7*s(q[2, 0])*s(q[5, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0]) - a7*s(q[4, 0])*s(q[5, 0])*c(q[0, 0])*c(q[2, 0]) + df*s(q[0, 0])*s(q[1, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + df*s(q[0, 0])*s(q[1, 0])*s(q[5, 0])*c(q[3, 0]) - df*s(q[0, 0])*s(q[2, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]) - df*s(q[0, 0])*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0]) + df*s(q[0, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - df*s(q[2, 0])*s(q[3, 0])*s(q[5, 0])*c(q[0, 0]) + df*s(q[2, 0])*c(q[0, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + df*s(q[4, 0])*c(q[0, 0])*c(q[2, 0])*c(q[5, 0]), 0], [0, -a4*c(q[1, 0])*c(q[2, 0]) - a5*s(q[1, 0])*s(q[3, 0]) - a5*c(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - a7*s(q[1, 0])*s(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - a7*s(q[1, 0])*s(q[5, 0])*c(q[3, 0]) + a7*s(q[2, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]) + a7*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[2, 0]) - a7*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) - d3*s(q[1, 0]) - d5*s(q[1, 0])*c(q[3, 0]) + d5*s(q[3, 0])*c(q[1, 0])*c(q[2, 0]) - df*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[4, 0]) + df*s(q[1, 0])*c(q[3, 0])*c(q[5, 0]) + df*s(q[2, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0]) - df*s(q[3, 0])*c(q[1, 0])*c(q[2, 0])*c(q[5, 0]) - df*s(q[5, 0])*c(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]), a4*s(q[1, 0])*s(q[2, 0]) + a5*s(q[1, 0])*s(q[2, 0])*c(q[3, 0]) - a7*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*s(q[5, 0]) + a7*s(q[1, 0])*s(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[4, 0])*c(q[2, 0])*c(q[5, 0]) - d5*s(q[1, 0])*s(q[2, 0])*s(q[3, 0]) + df*s(q[1, 0])*s(q[2, 0])*s(q[3, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[2, 0])*s(q[5, 0])*c(q[3, 0])*c(q[4, 0]) + df*s(q[1, 0])*s(q[4, 0])*s(q[5, 0])*c(q[2, 0]), a5*s(q[1, 0])*s(q[3, 0])*c(q[2, 0]) + a5*c(q[1, 0])*c(q[3, 0]) + a7*s(q[1, 0])*s(q[3, 0])*c(q[2, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[1, 0]) + a7*c(q[1, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + d5*s(q[1, 0])*c(q[2, 0])*c(q[3, 0]) - d5*s(q[3, 0])*c(q[1, 0]) + df*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[2, 0])*c(q[4, 0]) - df*s(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) + df*s(q[3, 0])*c(q[1, 0])*c(q[5, 0]) + df*s(q[5, 0])*c(q[1, 0])*c(q[3, 0])*c(q[4, 0]), a7*s(q[1, 0])*s(q[2, 0])*c(q[4, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[4, 0])*c(q[2, 0])*c(q[3, 0])*c(q[5, 0]) - a7*s(q[3, 0])*s(q[4, 0])*c(q[1, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[2, 0])*s(q[5, 0])*c(q[4, 0]) + df*s(q[1, 0])*s(q[4, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0]) - df*s(q[3, 0])*s(q[4, 0])*s(q[5, 0])*c(q[1, 0]), -a7*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*s(q[5, 0]) + a7*s(q[1, 0])*s(q[3, 0])*c(q[2, 0])*c(q[5, 0]) + a7*s(q[1, 0])*s(q[5, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0]) - a7*s(q[3, 0])*s(q[5, 0])*c(q[1, 0])*c(q[4, 0]) + a7*c(q[1, 0])*c(q[3, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[2, 0])*s(q[4, 0])*c(q[5, 0]) + df*s(q[1, 0])*s(q[3, 0])*s(q[5, 0])*c(q[2, 0]) - df*s(q[1, 0])*c(q[2, 0])*c(q[3, 0])*c(q[4, 0])*c(q[5, 0]) + df*s(q[3, 0])*c(q[1, 0])*c(q[4, 0])*c(q[5, 0]) + df*s(q[5, 0])*c(q[1, 0])*c(q[3, 0]), 0]])
