
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from scipy import integrate
from math import exp
from typing import Union, Tuple
from numba import njit

import mappings
import rmp_node
from rmp_leaf import LeafBase
import fabric



def attractor_test():
    xg = np.array([[2, 3]]).T
    m_u = 2
    m_l = 0.2
    alpha_m = 0.75
    k = 10
    alpha_psi = 10
    k_d = 10
    X0 = np.array([0, 0, 0.1, 0.0])
    time_interval = 0.01
    time_span = 100
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)

    attractor = fabric.GoalAttractor(m_u, m_l, alpha_m, k, alpha_psi, k_d)

    def dX(t, X):
        x = X[:2].reshape(-1, 1)
        x_dot = X[2:].reshape(-1, 1)
        M, F, _, _, _ = attractor.calc_fabric(x, x_dot, xg)
        a = LA.pinv(M) @ F
        return np.ravel(np.concatenate([x_dot, a]))

    sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
    print(sol.message)

    xi_s, pi_s, d_s, f_s, M_s = [], [], [], [], []
    xi_n_s, pi_n_s, d_n_s, f_n_s = [], [], [], []
    Le_s = []
    for i in range(len(sol.t)):
        x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
        M, F, xi, pi, d = attractor.calc_fabric(x, x_dot, xg)
        M_s.append(M)
        Le_s.append((x_dot.T @ M @ x_dot)[0,0])
        xi_s.append(xi)
        pi_s.append(pi)
        d_s.append(d)
        f_s.append(F)
        xi_n_s.append(LA.norm(xi))
        pi_n_s.append(LA.norm(pi))
        d_n_s.append(LA.norm(d))
        f_n_s.append(LA.norm(F))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol.y[0], sol.y[1], label="line")
    ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.grid(); ax.set_aspect('equal'); ax.legend()
    fig.savefig("fabric_test.png")

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 12 ))
    axes[0].plot(sol.t, sol.y[0], label="x")
    axes[0].plot(sol.t, sol.y[1], label="y")
    axes[1].plot(sol.t, sol.y[2], label="dx")
    axes[1].plot(sol.t, sol.y[3], label="dy")
    axes[2].plot(sol.t, xi_n_s, label="xi")
    axes[2].plot(sol.t, pi_n_s, label="pi")
    axes[2].plot(sol.t, d_n_s, label="d")
    axes[2].plot(sol.t, f_n_s, label="f = pi - xi - d")
    axes[3].plot(sol.t, Le_s, label="geometric Le")
    for ax in axes.ravel():
        ax.legend()
        ax.grid()

    fig.savefig("fabric_test_state.png")



    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_x = max(sol.y[0]); min_x = min(sol.y[0])
    max_y = max(sol.y[1]); min_y = min(sol.y[1])
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    max_range = max(max_x-min_x, max_y-min_y) * 0.5

    scale = 10
    f_scale = 0.1

    def update(i):
        ax.cla()
        ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        #ax.plot(sol.y[0][:i], sol.y[1][:i])

        
        eigvals, eigvecs = LA.eig(M_s[i])  # 計量の固有値と固有ベクトルを計算
        if np.any(np.iscomplex(eigvals)) or np.any(eigvals <= 1e-3): # not正定対称．リーマンじゃないのでスキップ
            met_axes_lengths = np.array([0, 0])
            met_angle = 0
        else:  # リーマン計量だから描写
            #print("riemman!")
            axes_lengths = 1.0 / np.sqrt(eigvals) * 0.1
            max_len = max(axes_lengths)
            #scale = min(2.0 / max_len, 1.0)
            met_axes_lengths = axes_lengths * scale
            met_angle = np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # 楕円の傾き

        c = patches.Ellipse(
            xy=(sol.y[0][i], sol.y[1][i]), 
            width = met_axes_lengths[0], height = met_axes_lengths[1],
            angle = met_angle,
            ec='k', fill=False
        )
        ax.add_patch(c)
        
        x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        xi = x + xi_s[i]*f_scale
        ax.plot([x[0,0], xi[0,0]], [x[1,0], xi[1,0]], label="xi")
        pi = x + pi_s[i]*f_scale
        ax.plot([x[0,0], pi[0,0]], [x[1,0], pi[1,0]], label="pi")
        f = x + f_s[i]*f_scale
        ax.plot([x[0,0], f[0,0]], [x[1,0], f[1,0]], label="f")
        
        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid()
        ax.set_aspect('equal')
        ax.legend()

    epoch_max = 60
    if len(sol.t) < epoch_max:
        step = 1
    else:
        step = len(sol.t) // epoch_max

    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(sol.t), step),
        interval=60
    )
    ani.save("fabrictest.gif", writer="pillow")

    plt.show()



def avoidance_test():
    r = 1
    k_b = 29
    alpha_b = 1
    alpha_sig = 50
    xo = np.array([[0, 0]]).T
    X0 = np.array([2, 0.1, -1, 0.0])
    time_interval = 0.01
    time_span = 40
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)

    obs = fabric.ObstacleAvoidance(r, k_b, alpha_b)

    def dX(t, X):
        x = X[:2].reshape(-1, 1)
        x_dot = X[2:].reshape(-1, 1)
        M, F, _, _, _, _, _ = obs.calc_fabric(x, x_dot, xo, np.zeros_like(xo))
        a = LA.pinv(M) @ F
        return np.ravel(np.concatenate([x_dot, a]))

    sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
    print(sol.message)

    xi_s, pi_s, d_s, f_s, M_s, F_s, m_s = [], [], [], [], [], [], []
    #xi_n_s, pi_n_s, d_n_s, f_n_s = [], [], [], []
    F_n_s = []
    Le_s = []
    for i in range(len(sol.t)):
        x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
        M, F, m, xi, pi, d, f = obs.calc_fabric(x, x_dot, xo, np.zeros_like(xo))
        M_s.append(M)
        F_s.append(F)
        Le_s.append(0)
        xi_s.append(xi)
        pi_s.append(pi)
        d_s.append(d)
        f_s.append(f)
        m_s.append(m)
        F_n_s.append(LA.norm(F))
        # xi_n_s.append(LA.norm(xi))
        # pi_n_s.append(LA.norm(pi))
        # d_n_s.append(LA.norm(d))
        # f_n_s.append(LA.norm(F))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol.y[0], sol.y[1], label="line")
    ax.scatter([xo[0,0]], [xo[1,0]], marker="+", color = "k", label="obs")
    c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=r, ec='k', fill=False)
    ax.add_patch(c)
    
    
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.grid(); ax.set_aspect('equal'); ax.legend()
    fig.savefig("fabric_test_obs.png")

    #print(xi_s)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 12 ))
    axes[0].plot(sol.t, sol.y[0], label="x")
    axes[0].plot(sol.t, sol.y[1], label="y")
    axes[1].plot(sol.t, sol.y[2], label="dx")
    axes[1].plot(sol.t, sol.y[3], label="dy")
    axes[2].plot(sol.t, xi_s, label="xi")
    axes[2].plot(sol.t, pi_s, label="pi")
    axes[2].plot(sol.t, d_s, label="d")
    axes[2].plot(sol.t, f_s, label="f = pi - xi - d")
    axes[2].plot(sol.t, m_s, label="m")
    axes[3].plot(sol.t, F_n_s, label="F")
    #axes[3].plot(sol.t, Le_s, label="geometric Le")
    for ax in axes.ravel():
        ax.legend()
        ax.grid()

    fig.savefig("fabric_test_obs_state.png")



    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_x = max(max(sol.y[0]), xo[0,0])
    min_x = min(min(sol.y[0]), xo[0,0])
    max_y = max(max(sol.y[1]), xo[1,0])
    min_y = min(min(sol.y[1]), xo[1,0])
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    max_range = max(max_x-min_x, max_y-min_y) * 0.5

    scale = 1000
    f_scale = 10

    def update(i):
        ax.cla()
        ax.scatter([xo[0,0]], [xo[1,0]], marker="+", color = "k", label="obs")
        c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=r, ec='k', fill=False)
        ax.add_patch(c)
        
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.plot(sol.y[0][:i], sol.y[1][:i])

        
        eigvals, eigvecs = LA.eig(M_s[i])  # 計量の固有値と固有ベクトルを計算
        print(eigvals)
        if np.any(np.iscomplex(eigvals)) or np.any(eigvals <= 1e-5): # not正定対称．リーマンじゃないのでスキップ
            met_axes_lengths = np.array([0, 0])
            met_angle = 0
        else:  # リーマン計量だから描写
            print("riemman!")
            axes_lengths = 1.0 / np.sqrt(eigvals) * 0.1
            max_len = max(axes_lengths)
            #scale = min(2.0 / max_len, 1.0)
            met_axes_lengths = axes_lengths * scale
            met_angle = np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # 楕円の傾き

        c = patches.Ellipse(
            xy=(sol.y[0][i], sol.y[1][i]), 
            width = met_axes_lengths[0], height = met_axes_lengths[1],
            angle = met_angle,
            ec='k', fill=False
        )
        ax.add_patch(c)
        
        x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        # xi = x + xi_s[i]*f_scale
        # ax.plot([x[0,0], xi[0,0]], [x[1,0], xi[1,0]], label="xi")
        # pi = x + pi_s[i]*f_scale
        # ax.plot([x[0,0], pi[0,0]], [x[1,0], pi[1,0]], label="pi")
        f = x + f_s[i]*f_scale
        ax.plot([x[0,0], f[0,0]], [x[1,0], f[1,0]], label="f")
        
        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid()
        ax.set_aspect('equal')
        ax.legend()

    epoch_max = 60
    if len(sol.t) < epoch_max:
        step = 1
    else:
        step = len(sol.t) // epoch_max

    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(sol.t), step),
        interval=60
    )
    ani.save("fabric_test_obs.gif", writer="pillow")

    plt.show()




if __name__ == "__main__":
    
    attractor_test()
    #avoidance_test()