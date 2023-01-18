
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import time
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from scipy import integrate
from math import exp, pi, cos, sin
from typing import Union, Tuple
from numba import njit

import mappings
import rmp_node
from rmp_leaf import LeafBase
import fabric
import multi_robot_rmp



# def print_progress(t, time_span, flag):
#     """sipy用プログレスバー"""
#     tmp = int(100 * t / time_span)
#     a, b = divmod(tmp, 10)
#     if b == 0 and flag != a:
#         print(tmp, "%")
#     return a

# 五角形の計算
r = 1
xs = []
ys = []
for i in range(5):
    xs.append(r * cos(2*pi/5 * i + pi/2))
    ys.append(r * sin(2*pi/5 * i + pi/2))



def test():
    """ロボット5台でテスト"""

    sim_name = "con"
    #sim_name = "pro"

    xg = np.array([[4, 4]]).T
    xo_s = [
        np.array([[1.0, 1.5]]).T,
        np.array([[2.0, 0.5]]).T,
        np.array([[2.5, 2.5]]).T,
    ]

    N = 5

    pres_pair = [
        [1, 4],
        [0, 2],
        [1, 3],
        [2, 4],
        [0, 3]
    ]  # 五角形

    # pres_pair = [
    #     [1, 2],
    #     [0, 2, 3],
    #     [0, 1, 4],
    #     [1],
    #     [2]
    # ]  #鶴翼の陣

    # pres_pair = [
    #     [1],
    #     [0, 2],
    #     [1, 3],
    #     [2, 4],
    #     [3]
    # ]  #鶴翼の陣


    # X0 = np.array([
    #     1, 1, 0, 0,
    #     0, 1, 0, 0,
    #     0, 0, 0, 0,
    #     1, -1, 0, 0,
    #     1, 0, 0, 0
    # ])


    # X0 = np.array([
    #     xs[0], ys[0], 0, 0,
    #     xs[1], ys[1], 0, 0,
    #     xs[2], ys[2], 0, 0,
    #     xs[3], ys[3], 0, 0,
    #     xs[4], ys[4], 0, 0
    # ]) + (np.random.rand(20) - 0.5)*0.9


    xu = 5; xl = 0
    yu = 5; yl = 0
    X0 = np.array([
        (xu-xl)*np.random.rand()+xl, (yu-yl)*np.random.rand()+yl, 0, 0,
        (xu-xl)*np.random.rand()+xl, (yu-yl)*np.random.rand()+yl, 0, 0,
        (xu-xl)*np.random.rand()+xl, (yu-yl)*np.random.rand()+yl, 0, 0,
        (xu-xl)*np.random.rand()+xl, (yu-yl)*np.random.rand()+yl, 0, 0,
        (xu-xl)*np.random.rand()+xl, (yu-yl)*np.random.rand()+yl, 0, 0
    ])


    # N = 3
    # pres_pair = [
    #     [1, 2],
    #     [2, 0],
    #     [1, 0]
    # ]
    # X0 = np.array([
    #     0, -1, 0, 0,
    #     0, 0, 0, 0,
    #     1, 0, 0, 0
    # ])



    ### フォーメーション維持 ###
    d = 0.5  #フォーメーション距離
    c = 1
    alpha = 5
    eta = 10
    pair_dis_pres = multi_robot_rmp.ParwiseDistancePreservation_a(d, c, alpha, eta)
    pair_dis_pres_fabric = fabric.ParwiseDistancePreservation(
        d=d, m_u=2, m_l=0.1, alpha_m=0.75, k=5, alpha_psi=1, k_d=10
    )


    # ロボット間の障害物回避
    Ds = 0.5
    pair_obs = multi_robot_rmp.PairwiseObstacleAvoidance(
        Ds=Ds, alpha=1e-5, eta=0.2, epsilon=1e-5
    )
    pair_obs_fabric = fabric.ObstacleAvoidance(
        r=Ds, k_b=20, alpha_b=0.75,
    )

    # 障害物回避
    obs_R = 0.5
    pair_obs_obs = multi_robot_rmp.PairwiseObstacleAvoidance(
        Ds=obs_R, alpha=10, eta=0.2, epsilon=1e-5
    )

    obs_fabric = fabric.ObstacleAvoidance(
        r=obs_R, k_b=20, alpha_b=0.75,
    )


    # 目標アトラクタ
    uni_attractor = multi_robot_rmp.UnitaryGoalAttractor_a(
        wu=10, wl=0.1, gain=150, sigma=1, alpha=1, tol=1e-3, eta=50
    )

    attractor_fabiic = fabric.GoalAttractor(
        m_u=2, m_l=0.2, alpha_m=0.75, k=150, alpha_psi=1, k_d=50
    )


    time_interval = 0.01
    time_span = 30
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)

    def dX(t, X):
        #print("t = ", t)
        X_dot = np.zeros((4*N, 1))
        x_s, x_dot_s = [], []
        for i in range(N):
            x_s.append(np.array([[X[4*i+0], X[4*i+1]]]).T)
            x_dot_s.append(np.array([[X[4*i+2], X[4*i+3]]]).T)

        for i in range(N):
            #print("i = ", i)
            root_M = np.zeros((2, 2))
            root_F = np.zeros((2, 1))

            # if i == 0:
            #     if sim_name == "con":
            #         M, F = uni_attractor.calc_rmp(x_s[i], x_dot_s[i], xg)
            #     elif sim_name == "pro":
            #         M, F, _, _, _ = attractor_fabiic.calc_fabric(x_s[i], x_dot_s[i], xg)
            #     else:
            #         assert False
            #     #print("Fat = ", F.T)
            #     root_M += M; root_F += F

            for j in range(N): #ロボット間の回避
                if i != j:
                    if sim_name == "con":
                        M, F = pair_obs.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                    elif sim_name =="pro":
                        M, F, _, _, _, _, _ = pair_obs_fabric.calc_fabric(x_s[i], x_dot_s[i], x_s[j])
                    else:
                        assert False
                    root_M += M; root_F += F

            # for xo in xo_s:  #障害物回避
            #     if sim_name == "con":
            #         M, F = pair_obs_obs.calc_rmp(x_s[i], x_dot_s[i], xo)
            #     elif sim_name == "pro":
            #         M, F, _, _, _, _, _ = obs_fabric.calc_fabric(x_s[i], x_dot_s[i], xo)
            #     else:
            #         assert False
            #     root_M += M; root_F += F

            for j in pres_pair[i]:  #フォーメーション維持
                if sim_name == "con":
                    M, F = pair_dis_pres.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                elif sim_name == "pro":
                    M, F = pair_dis_pres_fabric.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                else:
                    assert False
                root_M += M; root_F += F

            a = LA.pinv(root_M) @ root_F# + np.random.rand(2, 1)*10
            X_dot[4*i+0:4*i+1+1, :] = x_dot_s[i]
            X_dot[4*i+2:4*i+3+1, :] = a
            #print("a = ", a.T)

        return np.ravel(X_dot)

    t0 = time.perf_counter()
    sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
    print(sol.message)
    print("time = ", time.perf_counter() - t0)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(N):
        ax.plot(sol.y[4*i], sol.y[4*i+1], label="r{0}".format(i))

    for j in range(N):
        for k in pres_pair[j]:
            frame_x = [sol.y[4*k][-1], sol.y[4*j][-1]]
            frame_y = [sol.y[4*k+1][-1], sol.y[4*j+1][-1]]
            ax.plot(frame_x, frame_y, color="k")

    ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
    for xo in xo_s:
        c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
        ax.add_patch(c)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.grid();ax.set_aspect('equal'); ax.legend()
    fig.savefig(sim_name + ".png")

    # xi_s, pi_s, d_s, f_s, M_s = [], [], [], [], []
    # xi_n_s, pi_n_s, d_n_s, f_n_s = [], [], [], []
    # Le_s = []
    # for i in range(len(sol.t)):
    #     x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
    #     x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
    #     M, F, xi, pi, d = attractor.calc_fabric(x, x_dot, xg)
    #     M_s.append(M)
    #     Le_s.append((x_dot.T @ M @ x_dot)[0,0])
    #     xi_s.append(xi)
    #     pi_s.append(pi)
    #     d_s.append(d)
    #     f_s.append(F)
    #     xi_n_s.append(LA.norm(xi))
    #     pi_n_s.append(LA.norm(pi))
    #     d_n_s.append(LA.norm(d))
    #     f_n_s.append(LA.norm(F))



    # # fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 12 ))
    # # axes[0].plot(sol.t, sol.y[0], label="x")
    # # axes[0].plot(sol.t, sol.y[1], label="y")
    # # axes[1].plot(sol.t, sol.y[2], label="dx")
    # # axes[1].plot(sol.t, sol.y[3], label="dy")
    # # axes[2].plot(sol.t, xi_n_s, label="xi")
    # # axes[2].plot(sol.t, pi_n_s, label="pi")
    # # axes[2].plot(sol.t, d_n_s, label="d")
    # # axes[2].plot(sol.t, f_n_s, label="f = pi - xi - d")
    # # axes[3].plot(sol.t, Le_s, label="geometric Le")
    # # for ax in axes.ravel():
    # #     ax.legend()
    # #     ax.grid()

    # # fig.savefig("fabric_test_state.png")



    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_all, y_all = [], []
    for i in range(N):
        x_all.extend (sol.y[4*i])
        y_all.extend(sol.y[4*i+1])

    x_all.append(xg[0,0]); y_all.append(xg[1,0])

    max_x = max(x_all)
    min_x = min(x_all)
    max_y = max(y_all)
    min_y = min(y_all)
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    max_range = max(max_x-min_x, max_y-min_y) * 0.5


    time_template = 'time = %.2f [s]'


    scale = 10
    f_scale = 0.1

    def update(i):
        ax.cla()
        ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
        for xo in xo_s:
            c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
            ax.add_patch(c)

        for j in range(N):

            ax.plot(sol.y[4*j][:i], sol.y[4*j+1][:i], label="r{0}".format(j))

        for j in range(N):
            for k in pres_pair[j]:
                frame_x = [sol.y[4*k][i], sol.y[4*j][i]]
                frame_y = [sol.y[4*k+1][i], sol.y[4*j+1][i]]
                ax.plot(frame_x, frame_y, color="k")

        ax.set_title(time_template % sol.t[i])
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        #ax.plot(sol.y[0][:i], sol.y[1][:i])


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

        # x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
        # xi = x + xi_s[i]*f_scale
        # ax.plot([x[0,0], xi[0,0]], [x[1,0], xi[1,0]], label="xi")
        # pi = x + pi_s[i]*f_scale
        # ax.plot([x[0,0], pi[0,0]], [x[1,0], pi[1,0]], label="pi")
        # f = x + f_s[i]*f_scale
        # ax.plot([x[0,0], f[0,0]], [x[1,0], f[1,0]], label="f")

        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid()
        ax.set_aspect('equal')
        ax.legend()

    epoch_max = 80
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
    ani.save(sim_name + ".gif", writer="pillow")

    plt.show()



if __name__ == "__main__":

    test()