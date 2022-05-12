import numpy as np
from math import cos, sin

import mappings

l1 = 1.0
l2 = 1.0
l3 = 1.0
l4 = 1.0

q_min = np.array([[
    (-3/4)*np.pi+(1/2)*np.pi,
    (-3/4)*np.pi,
    (-3/4)*np.pi,
    (-3/4)*np.pi
]]).T

q_max = np.array([[
    (3/4)*np.pi+(1/2)*np.pi,
    (3/4)*np.pi,
    (3/4)*np.pi,
    (3/4)*np.pi
]]).T

q_neutral = np.array([[
    1/2*np.pi,
    0,
    0,
    0,
]]).T


class X1(mappings.Id):
    def phi(self, q):
        return np.array([
            [l1*cos(q[0,0])],
            [l1*sin(q[0,0])],
        ])

    def J(self, q):
        return np.array([
            [-l1*sin(q[0,0]), 0, 0, 0],
            [l1*cos(q[0,0]), 0, 0, 0]
        ])

    def J_dot(self, q, dq):
        return np.array([
            [-dq[0,0]*l1*cos(q[0,0]), 0, 0, 0],
            [-dq[0,0]*l1*sin(q[0,0]), 0, 0, 0]
        ])


class X2(mappings.Id):

    def phi(self, q):
        return np.array([
            [l1*cos(q[0,0]) + l2*cos(q[0,0] + q[1,0])],
            [l1*sin(q[0,0]) + l2*sin(q[0,0] + q[1,0])],
        ])

    def J(self, q):
        return np.array([
            [-l1*sin(q[0,0]) - l2*sin(q[0,0] + q[1,0]), -l2*sin(q[0,0] + q[1,0]), 0, 0],
            [l1*cos(q[0,0]) + l2*cos(q[0,0] + q[1,0]), l2*cos(q[0,0] + q[1,0]), 0, 0],
        ])

    def J_dot(self, q, dq):
        return np.array([
            [-dq[0,0]*l1*cos(q[0,0]) - l2*(dq[0,0] + dq[1,0])*cos(q[0,0] + q[1,0]), -l2*(dq[0,0] + dq[1,0])*cos(q[0,0] + q[1,0]), 0, 0],
            [-dq[0,0]*l1*sin(q[0,0]) - l2*(dq[0,0] + dq[1,0])*sin(q[0,0] + q[1,0]), -l2*(dq[0,0] + dq[1,0])*sin(q[0,0] + q[1,0]), 0, 0]
        ])


class X3(mappings.Id):
    def phi(self, q):
        return np.array([
            [l1*cos(q[0,0]) + l2*cos(q[0,0] + q[1,0]) + l3*cos(q[0,0] + q[1,0] + q[2,0])],
            [l1*sin(q[0,0]) + l2*sin(q[0,0] + q[1,0]) + l3*sin(q[0,0] + q[1,0] + q[2,0])],
        ])

    def J(self, q):
        return np.array([
            [-l1*sin(q[0,0]) - l2*sin(q[0,0] + q[1,0]) - l3*sin(q[0,0] + q[1,0] + q[2,0]), -l2*sin(q[0,0] + q[1,0]) - l3*sin(q[0,0] + q[1,0] + q[2,0]), -l3*sin(q[0,0] + q[1,0] + q[2,0]), 0],
            [l1*cos(q[0,0]) + l2*cos(q[0,0] + q[1,0]) + l3*cos(q[0,0] + q[1,0] + q[2,0]), l2*cos(q[0,0] + q[1,0]) + l3*cos(q[0,0] + q[1,0] + q[2,0]), l3*cos(q[0,0] + q[1,0] + q[2,0]), 0]
        ])

    def J_dot(self, q, dq):
        return np.array([
            [-dq[0,0]*l1*cos(q[0,0]) - l2*(dq[0,0] + dq[1,0])*cos(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*cos(q[0,0] + q[1,0] + q[2,0]), -l2*(dq[0,0] + dq[1,0])*cos(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*cos(q[0,0] + q[1,0] + q[2,0]), -l3*(dq[0,0] + dq[1,0] + dq[2,0])*cos(q[0,0] + q[1,0] + q[2,0]), 0],
            [-dq[0,0]*l1*sin(q[0,0]) - l2*(dq[0,0] + dq[1,0])*sin(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*sin(q[0,0] + q[1,0] + q[2,0]), -l2*(dq[0,0] + dq[1,0])*sin(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*sin(q[0,0] + q[1,0] + q[2,0]), -l3*(dq[0,0] + dq[1,0] + dq[2,0])*sin(q[0,0] + q[1,0] + q[2,0]), 0]
        ])


class X4(mappings.Id):
    def phi(self, q):
            return np.array([
                [l1*cos(q[0,0]) + l2*cos(q[0,0] + q[1,0]) + l3*cos(q[0,0] + q[1,0] + q[2,0]) + l4*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0])],
                [l1*sin(q[0,0]) + l2*sin(q[0,0] + q[1,0]) + l3*sin(q[0,0] + q[1,0] + q[2,0]) + l4*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0])],
            ])

    def J(self, q):
        return np.array([
            [-l1*sin(q[0,0]) - l2*sin(q[0,0] + q[1,0]) - l3*sin(q[0,0] + q[1,0] + q[2,0]) - l4*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l2*sin(q[0,0] + q[1,0]) - l3*sin(q[0,0] + q[1,0] + q[2,0]) - l4*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l3*sin(q[0,0] + q[1,0] + q[2,0]) - l4*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l4*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0])],
            [l1*cos(q[0,0]) + l2*cos(q[0,0] + q[1,0]) + l3*cos(q[0,0] + q[1,0] + q[2,0]) + l4*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0]), l2*cos(q[0,0] + q[1,0]) + l3*cos(q[0,0] + q[1,0] + q[2,0]) + l4*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0]), l3*cos(q[0,0] + q[1,0] + q[2,0]) + l4*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0]), l4*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0])],
        ])

    def J_dot(self, q, dq):
        return np.array([
            [-dq[0,0]*l1*cos(q[0,0]) - l2*(dq[0,0] + dq[1,0])*cos(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*cos(q[0,0] + q[1,0] + q[2,0]) - l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l2*(dq[0,0] + dq[1,0])*cos(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*cos(q[0,0] + q[1,0] + q[2,0]) - l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l3*(dq[0,0] + dq[1,0] + dq[2,0])*cos(q[0,0] + q[1,0] + q[2,0]) - l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*cos(q[0,0] + q[1,0] + q[2,0] + q[3,0])],
            [-dq[0,0]*l1*sin(q[0,0]) - l2*(dq[0,0] + dq[1,0])*sin(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*sin(q[0,0] + q[1,0] + q[2,0]) - l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l2*(dq[0,0] + dq[1,0])*sin(q[0,0] + q[1,0]) - l3*(dq[0,0] + dq[1,0] + dq[2,0])*sin(q[0,0] + q[1,0] + q[2,0]) - l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l3*(dq[0,0] + dq[1,0] + dq[2,0])*sin(q[0,0] + q[1,0] + q[2,0]) - l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0]), -l4*(dq[0,0] + dq[1,0] + dq[2,0] + dq[3,0])*sin(q[0,0] + q[1,0] + q[2,0] + q[3,0])]
        ])


if __name__ == "__main":
    pass