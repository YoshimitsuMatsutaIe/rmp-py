
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
import datetime
import os
import yaml
import shutil

import mappings
import rmp_node
from rmp_leaf import LeafBase
import fabric
import multi_robot_rmp
from multiprocessing import Pool, cpu_count


# def print_progress(t, time_span, flag):
#     """sipy用プログレスバー"""
#     tmp = int(100 * t / time_span)
#     a, b = divmod(tmp, 10)
#     if b == 0 and flag != a:
#         print(tmp, "%")
#     return a

# # 五角形の計算
# r = 1
# xs = []
# ys = []
# for i in range(5):
#     xs.append(r * cos(2*pi/5 * i + pi/2))
#     ys.append(r * sin(2*pi/5 * i + pi/2))



def test(exp_name, sim_param_path, i, rand):
    """ロボット5台でテスト"""
    
    data_label = str(i)
    dir_base = "../syuron/formation_preservation_only/" + exp_name + "/"
    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(dir_base + "csv", exist_ok=True)
    os.makedirs(dir_base + "fig", exist_ok=True)
    os.makedirs(dir_base + "animation", exist_ok=True)
    os.makedirs(dir_base + "config", exist_ok=True)
    os.makedirs(dir_base + "message", exist_ok=True)
    
    #shutil.copy2(sim_param_path, self.dir_base)
    with open(sim_param_path) as f:
        sim_param = yaml.safe_load(f)
    with open(dir_base + "config/" + data_label + '.yaml', 'w') as f:
        yaml.dump(sim_param, f)

    # xg = np.array([[4, 4]]).T
    # xo_s = [
    #     np.array([[1.0, 1.5]]).T,
    #     np.array([[2.0, 0.5]]).T,
    #     np.array([[2.5, 2.5]]).T,
    # ]
    # xg = sim_param["goal"]
    # xo_s = sim_param["obstacle"]
    xg = None
    xo_s = None

    N = sim_param["N"]
    pres_pair = sim_param["pair"]

    # pres_pair = [
    #     [1, 4],
    #     [0, 2],
    #     [1, 3],
    #     [2, 4],
    #     [0, 3]
    # ]  # 五角形

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


    xu = sim_param["initial_condition"]["value"]["x_max"]
    xl = sim_param["initial_condition"]["value"]["x_min"]
    yu = sim_param["initial_condition"]["value"]["y_max"]
    yl = sim_param["initial_condition"]["value"]["y_min"]
    X0_ = []
    for _ in range(N):
        X0_.extend([
            rand.uniform(xl, xu),
            rand.uniform(yl, yu),
            0,
            0,
        ])
    X0 = np.array(X0_)
    #print(X0)


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


    rmp = sim_param["controller"]["rmp"]
    fab = sim_param["controller"]["fabric"]
    ### フォーメーション維持 ###
    formation_rmp = multi_robot_rmp.ParwiseDistancePreservation_a(**rmp["formation_preservation"])
    formation_fab = fabric.ParwiseDistancePreservation(**fab["formation_preservation"])

    # ロボット間の障害物回避
    pair_avoidance_rmp = multi_robot_rmp.PairwiseObstacleAvoidance(**rmp["pair_avoidance"])
    pair_avoidance_fab = fabric.ObstacleAvoidance(**fab["pair_avoidance"])

    # 障害物回避
    obs_avoidance_rmp = multi_robot_rmp.PairwiseObstacleAvoidance(**rmp["obstacle_avoidance"])
    obs_avoidamce_fab = fabric.ObstacleAvoidance(**fab["pair_avoidance"])
    obs_R = rmp["obstacle_avoidance"]["Ds"]

    # 目標アトラクタ
    attractor_rmp = multi_robot_rmp.UnitaryGoalAttractor_a(**rmp["goal_attractor"])
    attractor_fab = fabric.GoalAttractor(**fab["goal_attractor"])

    time_interval = sim_param["time_interval"]
    time_span = sim_param["time_span"]
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)


    def dX(t, X, sim_name):
        #print("t = ", t)
        X_dot = np.zeros((4*N, 1))
        x_s, x_dot_s = [], []
        for i in range(N):
            x_s.append(np.array([[X[4*i+0], X[4*i+1]]]).T)
            x_dot_s.append(np.array([[X[4*i+2], X[4*i+3]]]).T)

        for i in range(N):
            root_M = np.zeros((2, 2))
            root_F = np.zeros((2, 1))

            if i == 0 and xg is not None:
                if sim_name == "rmp":
                    M, F = attractor_rmp.calc_rmp(x_s[i], x_dot_s[i], xg)
                elif sim_name == "fabric":
                    M, F, _, _, _ = attractor_fab.calc_fabric(x_s[i], x_dot_s[i], xg)
                else:
                    assert False
                #print("Fat = ", F.T)
                root_M += M; root_F += F

            for j in range(N): #ロボット間の回避
                if i != j:
                    if sim_name == "rmp":
                        M, F = pair_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                    elif sim_name =="fabric":
                        M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(x_s[i], x_dot_s[i], x_s[j])
                    else:
                        assert False
                    root_M += M; root_F += F

            if xo_s is not None:
                for xo in xo_s:  #障害物回避
                    if sim_name == "rmp":
                        M, F = obs_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], xo)
                    elif sim_name == "fabric":
                        M, F, _, _, _, _, _ = obs_avoidamce_fab.calc_fabric(x_s[i], x_dot_s[i], xo)
                    else:
                        assert False
                    root_M += M; root_F += F

            for j in pres_pair[i]:  #フォーメーション維持
                if sim_name == "rmp":
                    M, F = formation_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                elif sim_name == "fabric":
                    M, F = formation_fab.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                else:
                    assert False
                root_M += M; root_F += F

            a = LA.pinv(root_M) @ root_F
            X_dot[4*i+0:4*i+1+1, :] = x_dot_s[i]
            X_dot[4*i+2:4*i+3+1, :] = a
            
        return np.ravel(X_dot)


    for sim_name in ["rmp", "fabric"]:
        #t0 = time.perf_counter()
        sol = integrate.solve_ivp(
            fun=dX, 
            t_span=tspan, 
            y0=X0, 
            t_eval=teval, 
            args=(sim_name,)
        )
        #print(sol.message)
        with open(dir_base + "message/" + data_label + "_" + sim_name + '.txt', 'w') as f:
            f.write(sol.message)
        #print("time = ", time.perf_counter() - t0)

        ## CSV保存
        # まずはヘッダーを準備
        header = "t"
        for i in range(N):
            header += ",x" + str(i)
        for i in range(N):
            header += ",dx" + str(i)

        # 時刻歴tと解xを一つのndarrayにする
        data = np.concatenate(
            [sol.t.reshape(1, len(sol.t)).T, sol.y.T],  # sol.tは1次元配列なので2次元化する
            axis=1
        )
        np.savetxt(
            dir_base + "csv/" + data_label + "_" + sim_name + '.csv',
            data,
            header = header,
            comments = '',
            delimiter = ","
        )


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.plot(sol.y[4*i], sol.y[4*i+1], label="r{0}".format(i))

        for j in range(N):
            for k in pres_pair[j]:
                frame_x = [sol.y[4*k][-1], sol.y[4*j][-1]]
                frame_y = [sol.y[4*k+1][-1], sol.y[4*j+1][-1]]
                ax.plot(frame_x, frame_y, color="k")

        # ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
        # for xo in xo_s:
        #     c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
        #     ax.add_patch(c)

        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.grid();ax.set_aspect('equal'); ax.legend()
        fig.savefig(dir_base + "fig/" + data_label + "_" + sim_name + ".png")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_all, y_all = [], []
        for i in range(N):
            x_all.extend (sol.y[4*i])
            y_all.extend(sol.y[4*i+1])

        #x_all.append(xg[0,0]); y_all.append(xg[1,0])

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
            # ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
            # for xo in xo_s:
            #     c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
            #     ax.add_patch(c)

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

        #t0 = time.perf_counter()
        ani = anm.FuncAnimation(
            fig = fig,
            func = update,
            frames = range(0, len(sol.t), step),
            interval=60
        )
        ani.save(dir_base + "animation/" + data_label+ sim_name + "_"  + '.gif', writer="pillow")

        #print("ani_time = ", time.perf_counter() - t0)


def runner(sim_path, n):
    date_now = datetime.datetime.now()
    data_label = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    dir_base = "../syuron/formation_preservation_only/" + data_label
    os.makedirs(dir_base, exist_ok=True)
    
    itr = [
        (data_label, sim_path, i, np.random.RandomState(np.random.randint(0, 10000000)))
        for i in range(n)
    ]
    
    
    core = cpu_count()
    #core = 2
    with Pool(core) as p:
        result = p.starmap(
            func = test,
            iterable = itr
        )


if __name__ == "__main__":
    sim_path = "/home/matsuta_conda/src/rmp-py/config_syuron/test1.yaml"
    runner(sim_path, 10)
