
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



@njit
def calc_Pe(M, q_dot):
    M_harf = M ** (1/2)
    print(M_harf)
    v = M_harf @ q_dot
    v_hat = v / LA.norm(v)
    Pe = M_harf @ (np.eye(2) - v_hat @ v_hat.T) @ LA.pinv(M_harf)
    return Pe


def simple_test():
    xg = np.array([[0, 0]]).T
    xo = np.array([[0.3, 0.0]]).T
    X0 = np.array([1, 0.01, -0., 0.0])
    
    time_interval = 0.01
    time_span = 60
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)

    attractor = fabric.GoalAttractor(
        m_u=2, m_l=0.2, alpha_m=0.75, k=10, alpha_psi=10, k_d=100
    )
    obs = fabric.ObstacleAvoidance(r=0.2, k_b=0.1, alpha_b=1)




    def dX(t, X):
        x = X[:2].reshape(-1, 1); x_dot = X[2:].reshape(-1, 1)
        root_M = np.zeros((2,2)); root_F = np.zeros((2,1))
        M, F, _, _, _ = attractor.calc_fabric(x, x_dot, xg)
        root_M += M; root_F += F
        M, F, _, _, _, _, _ = obs.calc_fabric(x, x_dot, xo, np.zeros_like(xo))
        root_M += M; root_F += F
        
        
        # Pe = calc_Pe(root_M, x_dot)
        # ene_F = Pe @ root_F
        # a = LA.pinv(root_M) @ ene_F
        
        a = LA.pinv(root_M) @ root_F
        old_a = a
        if LA.norm(x_dot) < 1e-5:
            print("error")
            x_dot_hat = x_dot / LA.norm(x_dot)
            #print((np.eye(2) - x_dot_hat @ x_dot_hat.T)@a)
            a = (np.eye(2) - x_dot_hat @ x_dot_hat.T) @ a
        #print("old = {0}, now = {1}".format(old_a.T, a.T))
        
        return np.ravel(np.concatenate([x_dot, a]))

    sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
    print(sol.message)


    g_xi, g_pi, g_d, g_f, g_M, g_Le = [], [], [], [], [], []
    g_xi_n, g_pi_n, g_d_n, g_f_n = [], [], [], []
    o_xi, o_pi, o_d, o_f, o_M, o_F, o_m = [], [], [], [], [], [], []
    o_F_n, o_Le = [], []
    goal_error, obs_error = [], []
    ene_a = []
    v = []
    for i in range(len(sol.t)):
        x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
        v.append(LA.norm(x_dot))
        M, F, xi, pi, d = attractor.calc_fabric(x, x_dot, xg)
        g_M.append(M)
        g_Le.append((x_dot.T @ M @ x_dot)[0,0])
        g_xi.append(xi)
        g_pi.append(pi)
        g_d.append(d)
        g_f.append(F)
        g_xi_n.append(LA.norm(xi))
        g_pi_n.append(LA.norm(pi))
        g_d_n.append(LA.norm(d))
        g_f_n.append(LA.norm(F))
        goal_error.append(LA.norm(xg - x))
        
        M, F, m, xi, pi, d, f = obs.calc_fabric(x, x_dot, xo, np.zeros_like(xo))
        o_M.append(M)
        o_F.append(F)
        o_Le.append(0)
        o_xi.append(xi)
        o_pi.append(pi)
        o_d.append(d)
        o_f.append(f)
        o_m.append(m)
        o_F_n.append(LA.norm(F))
        obs_error.append(LA.norm(xo-x))
        
        
        
        

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol.y[0], sol.y[1], label="line")
    ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.grid(); ax.set_aspect('equal'); ax.legend()
    fig.savefig("fabric_test.png")
    

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 12 ))
    axes[0].plot(sol.t, goal_error, label="g errpr")
    axes[0].plot(sol.t, obs_error, label="o error")
    axes[1].plot(sol.t, v, label="dx")

    axes[2].plot(sol.t, g_xi_n, label="goal xi")
    axes[2].plot(sol.t, g_pi_n, label="goal pi")
    axes[2].plot(sol.t, g_d_n, label="goal d")
    axes[2].plot(sol.t, g_f_n, label="goal f = pi - xi - d")
    #axes[3].plot(sol.t, g_Le, label="geometric Le")
    
    axes[3].plot(sol.t, o_xi, label="obs  xi")
    axes[3].plot(sol.t, o_pi, label="obs  pi")
    axes[3].plot(sol.t, o_d, label="obs  d")
    axes[3].plot(sol.t, o_f, label="obs  f = pi - xi - d")
    axes[4].plot(sol.t, o_m, label="obs m")
    axes[4].plot(sol.t, o_F_n, label="obs F")
    
    
    
    for ax in axes.ravel():
        ax.legend()
        ax.grid()

    fig.savefig("simple_test_state.png")



    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_x = max(max(sol.y[0]), xg[0,0], xo[0,0])
    min_x = min(min(sol.y[0]), xg[0,0], xo[0,0])
    max_y = max(max(sol.y[1]), xg[1,0], xo[1,0])
    min_y = min(min(sol.y[1]), xg[1,0], xo[1,0])
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    max_range = max(max_x-min_x, max_y-min_y) * 0.5 * 1.2

    scale = 10
    f_scale = 0.05

    def update(i):
        ax.cla()
        ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
        ax.scatter([xo[0,0]], [xo[1,0]], marker="+", color = "k", label="obs")
        c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs.r, ec='k', fill=False)
        ax.add_patch(c)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.plot(sol.y[0][:i], sol.y[1][:i])
        ax.scatter([sol.y[0][i]], [sol.y[1][i]])

        # eigvals, eigvecs = LA.eig(M_s[i])  # 計量の固有値と固有ベクトルを計算
        # if np.any(np.iscomplex(eigvals)) or np.any(eigvals <= 1e-3): # not正定対称．リーマンじゃないのでスキップ
        #     met_axes_lengths = np.array([0, 0])
        #     met_angle = 0
        # else:  # リーマン計量だから描写
        #     #print("riemman!")
        #     axes_lengths = 1.0 / np.sqrt(eigvals) * 0.1
        #     max_len = max(axes_lengths)
        #     #scale = min(2.0 / max_len, 1.0)
        #     met_axes_lengths = axes_lengths * scale
        #     met_angle = np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # 楕円の傾き

        # c = patches.Ellipse(
        #     xy=(sol.y[0][i], sol.y[1][i]), 
        #     width = met_axes_lengths[0], height = met_axes_lengths[1],
        #     angle = met_angle,
        #     ec='k', fill=False
        # )
        # ax.add_patch(c)
        
        x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        xi = x + g_xi[i]*f_scale
        ax.plot([x[0,0], xi[0,0]], [x[1,0], xi[1,0]], label="goal xi")
        pi = x + g_pi[i]*f_scale
        ax.plot([x[0,0], pi[0,0]], [x[1,0], pi[1,0]], label="goal pi")
        f = x + g_f[i]*f_scale
        ax.plot([x[0,0], f[0,0]], [x[1,0], f[1,0]], label="goal f")
        
        f = x + o_F[i]*f_scale
        ax.plot([x[0,0], f[0,0]], [x[1,0], f[1,0]], label="obs  F")
        
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
        interval=80
    )
    ani.save("simple_test.gif", writer="pillow")
    ani.save("simple_test.mp4", writer="ffmpeg")

    plt.show()

    return


# def avoidance_test():
#     r = 1
#     k_b = 29
#     alpha_b = 1
#     alpha_sig = 50
#     xo = np.array([[0, 0]]).T
#     X0 = np.array([2, 0.1, -1, 0.0])
#     time_interval = 0.01
#     time_span = 40
#     tspan = (0, time_span)
#     teval = np.arange(0, time_span, time_interval)

#     obs = fabric.ObstacleAvoidance(r=1, k_b=29, alpha_b=1)

#     def dX(t, X):
#         x = X[:2].reshape(-1, 1)
#         x_dot = X[2:].reshape(-1, 1)
#         M, F, _, _, _, _, _ = obs.calc_fabric(x, x_dot, xo, np.zeros_like(xo))
#         a = LA.pinv(M) @ F
#         return np.ravel(np.concatenate([x_dot, a]))

#     sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
#     print(sol.message)

#     xi_s, pi_s, d_s, f_s, M_s, F_s, m_s = [], [], [], [], [], [], []
#     #xi_n_s, pi_n_s, d_n_s, f_n_s = [], [], [], []
#     F_n_s = []
#     Le_s = []
#     for i in range(len(sol.t)):
#         x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
#         x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
#         M, F, m, xi, pi, d, f = obs.calc_fabric(x, x_dot, xo, np.zeros_like(xo))
#         M_s.append(M)
#         F_s.append(F)
#         Le_s.append(0)
#         xi_s.append(xi)
#         pi_s.append(pi)
#         d_s.append(d)
#         f_s.append(f)
#         m_s.append(m)
#         F_n_s.append(LA.norm(F))
#         # xi_n_s.append(LA.norm(xi))
#         # pi_n_s.append(LA.norm(pi))
#         # d_n_s.append(LA.norm(d))
#         # f_n_s.append(LA.norm(F))

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(sol.y[0], sol.y[1], label="line")
#     ax.scatter([xo[0,0]], [xo[1,0]], marker="+", color = "k", label="obs")
#     c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=r, ec='k', fill=False)
#     ax.add_patch(c)
    
    
#     ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
#     ax.grid(); ax.set_aspect('equal'); ax.legend()
#     fig.savefig("fabric_test_obs.png")

#     #print(xi_s)

#     fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 12 ))
#     axes[0].plot(sol.t, sol.y[0], label="x")
#     axes[0].plot(sol.t, sol.y[1], label="y")
#     axes[1].plot(sol.t, sol.y[2], label="dx")
#     axes[1].plot(sol.t, sol.y[3], label="dy")
#     axes[2].plot(sol.t, xi_s, label="xi")
#     axes[2].plot(sol.t, pi_s, label="pi")
#     axes[2].plot(sol.t, d_s, label="d")
#     axes[2].plot(sol.t, f_s, label="f = pi - xi - d")
#     axes[2].plot(sol.t, m_s, label="m")
#     axes[3].plot(sol.t, F_n_s, label="F")
#     #axes[3].plot(sol.t, Le_s, label="geometric Le")
#     for ax in axes.ravel():
#         ax.legend()
#         ax.grid()

#     fig.savefig("fabric_test_obs_state.png")



#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     max_x = max(max(sol.y[0]), xo[0,0])
#     min_x = min(min(sol.y[0]), xo[0,0])
#     max_y = max(max(sol.y[1]), xo[1,0])
#     min_y = min(min(sol.y[1]), xo[1,0])
#     mid_x = (max_x + min_x) * 0.5
#     mid_y = (max_y + min_y) * 0.5
#     max_range = max(max_x-min_x, max_y-min_y) * 0.5

#     scale = 1000
#     f_scale = 5

#     def update(i):
#         ax.cla()
#         ax.scatter([xo[0,0]], [xo[1,0]], marker="+", color = "k", label="obs")
#         c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=r, ec='k', fill=False)
#         ax.add_patch(c)
        
#         ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
#         ax.plot(sol.y[0][:i], sol.y[1][:i])

        
#         eigvals, eigvecs = LA.eig(M_s[i])  # 計量の固有値と固有ベクトルを計算
#         print(eigvals)
#         if np.any(np.iscomplex(eigvals)) or np.any(eigvals <= 1e-5): # not正定対称．リーマンじゃないのでスキップ
#             met_axes_lengths = np.array([0, 0])
#             met_angle = 0
#         else:  # リーマン計量だから描写
#             print("riemman!")
#             axes_lengths = 1.0 / np.sqrt(eigvals) * 0.1
#             max_len = max(axes_lengths)
#             #scale = min(2.0 / max_len, 1.0)
#             met_axes_lengths = axes_lengths * scale
#             met_angle = np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # 楕円の傾き

#         c = patches.Ellipse(
#             xy=(sol.y[0][i], sol.y[1][i]), 
#             width = met_axes_lengths[0], height = met_axes_lengths[1],
#             angle = met_angle,
#             ec='k', fill=False
#         )
#         ax.add_patch(c)
        
#         x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
#         # xi = x + xi_s[i]*f_scale
#         # ax.plot([x[0,0], xi[0,0]], [x[1,0], xi[1,0]], label="xi")
#         # pi = x + pi_s[i]*f_scale
#         # ax.plot([x[0,0], pi[0,0]], [x[1,0], pi[1,0]], label="pi")
#         f = x + f_s[i]*f_scale
#         ax.plot([x[0,0], f[0,0]], [x[1,0], f[1,0]], label="f")
        
#         ax.set_xlim(mid_x-max_range, mid_x+max_range)
#         ax.set_ylim(mid_y-max_range, mid_y+max_range)
#         ax.grid()
#         ax.set_aspect('equal')
#         ax.legend()

#     epoch_max = 60
#     if len(sol.t) < epoch_max:
#         step = 1
#     else:
#         step = len(sol.t) // epoch_max

#     ani = anm.FuncAnimation(
#         fig = fig,
#         func = update,
#         frames = range(0, len(sol.t), step),
#         interval=60
#     )
#     ani.save("fabric_test_obs.gif", writer="pillow")

#     plt.show()




if __name__ == "__main__":
    
    simple_test()
