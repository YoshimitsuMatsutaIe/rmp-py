"""クアッドコプターのダイナミクス"""


import numpy as np
from math import cos, sin, pi, tan
from numba import njit

from scipy import integrate
import matplotlib.pyplot as plt

#@njit
def dynamics(state, state_d, ddq_d):
    g = 9.8  # 重力加速度
    l = 0.23
    m = 0.52
    Ixx = 0.0075; Iyy = Ixx; Izz = 0.013
    mu = 0.01  # 粘性抵抗
    
    phi, phi_dot, theta, theta_dot, psi, psi_dot, \
        z, z_dot, x, x_dot, y, y_dot = state
    phi_d, phi_dot_d, theta_d, theta_dot_d, psi_d, psi_dot_d, \
        z_d, z_dot_d, x_d, x_dot_d, y_d, y_dot_d = state_d
    phi_ddot, theta_ddot, psi_ddot, z_ddot, x_ddot, y_ddot = ddq_d
    # a1 = (Iyy - Izz) / Ixx
    # a3 = (Izz - Ixx) / Iyy
    # a5 = (Ixx - Iyy) / Izz
    # b1 = l / Ixx
    # b2 = l / Iyy
    # b3 = l / Izz

    a1=-0.733
    a3=0.733
    a5=0
    b1=30.67
    b2=30.67
    b3=17.69
    
    # T = np.array([
    #     [b, b, b, b],
    #     [-b, b, 0., 0.],
    #     [0., 0., -b, b],
    #     [d, d, -d, -d]
    # ])
    # OMEGA_2 = np.array([
    #     [omega[0]**2],
    #     [omega[1]**2],
    #     [omega[2]**2],
    #     [omega[3]**2],
    # ])
    
    # u = T @ OMEGA_2

    # バックステッピング法
    c11 = 265  #1 - c7**2
    c12 = 60  #c7 + c8
    cx1 = 500  #1 - c9**2
    cx2 = 1000  #c9 + c10
    c21 = 800
    c31 = c21
    c41 = c21
    c22 = 60
    c32 = c22
    c42 = c22
    c7 = 3.5
    c9 = 2
    cy1 = cx1#1 - c11**2
    cy2 = cx2#c11 + c12
    
    e1 = phi_d - phi
    e2 = phi_dot_d - phi_dot
    e3 = theta_d - theta
    e4 = theta_dot_d - theta_dot
    e5 = psi_d - psi
    e6 = psi_dot_d - psi_dot
    
    e7 = z_d - z
    e8 = c7*e7 + z_dot_d - z_dot
    e9 = x_d - x
    e11 = y_d - y
    e10 = c9*e9 + x_dot_d - x_dot
    e12 = c11*e11 + y_dot_d - y_dot
    u1 = m / (cos(phi)*cos(theta)+1e-5) * (g + c11*e7 + c12*e8)
    ux = m * (cx1*e9 + cx2*e10)
    uy = -m * (cy1*e11 + cy2*e12)
    u2 = 1/b1 * (c21*e1 + c22*e2 + phi_ddot - theta_dot*psi_dot*a1 + uy)
    u3 = 1/b2 * (c31*e3 + c32*e4 + theta_ddot - phi_dot*psi_dot*a3 + ux)
    u4 = 1/b3 * (c41*e5 + c42*e6 + psi_ddot - theta_dot*phi_dot*a5)

    #u1 = 0; u2 = 0; u3 = 0; u4 = 0

    print("u = {0}, {1}, {2}, {3}".format(u1, u2, u3, u4))
    return np.array([
        phi_dot,
        theta_dot*psi_dot*a1 + b1*u2,
        theta_dot,
        phi_dot*psi_dot*a3 + b2*u3,
        psi_dot,
        phi_dot*theta_dot*a5 + b3*u4,
        z_dot,
        cos(phi)*cos(theta)*u1/m - g - mu*z_dot,
        x_dot,
        -mu*x_dot + (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))/m*u1,
        y_dot, 
        -mu*y_dot + (cos(phi)*sin(theta)*cos(psi) + sin(phi)*cos(psi))/m*u1,
    ])


# @njit
# def pd_controller(state, kp, kd):
    
    


# class Dynamics:
#     def __init__(
#         self,
#         m=0.52, mu=0.01,
#         a1=-0.733, a2=0.733, a3=0,
#         b1=30.67, b2=30.67, b3=17.69,
#         g=9.8
#     ):
#         self.m = m
#         self.mu = mu
#         self.a1 = a1
#         self.a2 = a2
#         self.a3 = a3
#         self.b1 = b1
#         self.b2 = b2
#         self.b3 = b3
#         self.g = g
    
#     def calc_dynamics(self, x, u):
#         return dynamics(
#             x, u, 
#             self.m, self.mu, 
#             self.a1, self.a2, self.a3, self.b1, self.b2, self.b3,
#             self.g
#             )




def dX(t, X):
    print("t = ", t)
    #print(X)
    state_d = np.array([
        0, 0, 0, 0, 0, 0,
        0.5, 0, 0.01, 0, 0, 0
    ])
    ddq_d = np.array([
        0, 0, 0,
        0, 0, 0
    ])
    x_dot = dynamics(X, state_d, ddq_d)
    return x_dot

if __name__ == "__main__":
    
    X0 = np.array([
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ])
    sol = integrate.solve_ivp(
        fun=dX,
        t_span=(0, 10),
        t_eval=np.arange(0, 10, 0.01),
        y0=X0
    )
    print(sol.message)
    
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(sol.t, sol.y[0], label="phi")
    axes[0].plot(sol.t, sol.y[1], label="phi_dot")
    axes[0].plot(sol.t, sol.y[2], label="theta")
    axes[0].plot(sol.t, sol.y[3], label="theta_dpt")
    axes[0].plot(sol.t, sol.y[4], label="psi")
    axes[0].plot(sol.t, sol.y[5], label="psi_dot")
    axes[1].plot(sol.t, sol.y[6], label="z")
    axes[1].plot(sol.t, sol.y[7], label="dz")
    axes[1].plot(sol.t, sol.y[8], label="x")
    axes[1].plot(sol.t, sol.y[9], label="dx")
    axes[1].plot(sol.t, sol.y[10], label="y")
    axes[1].plot(sol.t, sol.y[11], label="dy")
    
    axes[0].legend()
    axes[1].legend()
    fig.savefig("coptor.png")
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(sol.y[8], sol.y[10], sol.y[6], label="tra")
    ax.scatter([0], [0], [1], label="goal", color="r")
    ax.legend()
    ax.set_aspect('equal')
    fig.savefig("coptor_3d.png")
    
    # X = np.array([
    #     0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 1, 0,
    # ])
    # state_d = np.array([
    #     0, 0, 0, 0, 0, 0,
    #     1, 0, 1, 0, 1, 0
    # ])
    # ddq_d = np.array([
    #     0, 0, 0,
    #     0, 0, 0
    # ])
    # x_dot = dynamics(X, state_d, ddq_d)
    # print(x_dot)