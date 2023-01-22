"""車モデル"""


import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import time
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from scipy import integrate
from math import exp, pi, cos, sin, tan
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

import config_syuron.car_1 as car_1


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

@njit("f8[:,:](f8)")
def rotate(theta):
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

@njit("f8[:,:](f8)")
def rotate_dot(theta):
    """theta微分"""
    return np.array([
        [-sin(theta), -cos(theta)],
        [cos(theta), -sin(theta)]
    ])


@njit
def car_ex(x, y, theta, dx, dy, omega, v, xi, L):
    beta = np.arctan(tan(xi) / 2)

    J =  np.array([
        [cos(theta), 0.0],
        [sin(theta), 0.0],
        [sin(2*beta)/L, (4*v*cos(2*beta))/(3*cos(xi)**2+1)]
    ])
    T = np.array([
        [0.0],
        [v**2/L * cos(theta) * sin(2*beta)],
        [0.0]
    ])
    
    return J, T

@njit
def turtle_bot(x, y, theta, dx, dy, omega, v, xi):
    J = np.array([
        [cos(theta), 0.0],
        [sin(theta), 0.0],
        [0.0, 1.0]
    ])
    T = np.array([
        [-v * sin(theta) * omega],
        [v * cos(theta) * omega],
        [0.0]
    ])
    
    return J, T

@njit("f8[:,:](f8[:,:], f8)")
def J_transform(x_bar, theta):
    return np.array([
        [1.0, 0.0, -sin(theta)*x_bar[0,0] - cos(theta)*x_bar[1,0]],
        [0.0, 1.0, cos(theta)*x_bar[0,0] - sin(theta)*x_bar[1,0]]
    ])


#@njit#("f8[:,:](f8)")
def calc_cpoint_state(x, x_dot, theta, omega, x_bars):
    y_s = []
    y_dot_s = []
    for i, x_bar in enumerate(x_bars):
        y_s.append(
            x + rotate(theta) @ x_bar
        )
        y_dot_s.append(
            x_dot + omega*rotate_dot(theta) @ x_bar
        )
    return y_s, y_dot_s


def test(exp_name, sim_param, i, rand):
    
    data_label = str(i)
    dir_base = "../syuron/formation_preservation_only/" + exp_name + "/"
    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(dir_base + "csv", exist_ok=True)
    os.makedirs(dir_base + "fig", exist_ok=True)
    os.makedirs(dir_base + "animation", exist_ok=True)
    os.makedirs(dir_base + "config", exist_ok=True)
    os.makedirs(dir_base + "message", exist_ok=True)
    os.makedirs(dir_base + "state", exist_ok=True)
    
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
    xg = np.array([[0.5, 0.5]]).T
    #xg = None
    xo_s = None

    N = sim_param["N"]
    robot_r = sim_param["robot_r"]
    sdim = 8
    pres_pair = sim_param["pair"]
    
    cpoint_num = sim_param["robot_cpoints_num"]  #ロボット一台あたりの制御点の数
    x_bar_s = []
    for j in range(cpoint_num):
        theta = 2*np.pi / cpoint_num * j
        x_bar_s.append(
            np.array([[robot_r*cos(theta), robot_r*sin(theta)]]).T
        )
    #print(x_bar_s)

    rmp = sim_param["controller"]["rmp"]
    fab = sim_param["controller"]["fabric"]
    
    ### フォーメーション維持 ###
    formation_rmp = multi_robot_rmp.ParwiseDistancePreservation_a(**rmp["formation_preservation"])
    formation_fab = fabric.ParwiseDistancePreservation(**fab["formation_preservation"])

    # ロボット間の障害物回避
    pair_avoidance_rmp = multi_robot_rmp.PairwiseObstacleAvoidance(**rmp["pair_avoidance"])
    pair_avoidance_fab = fabric.ObstacleAvoidance(**fab["pair_avoidance"])
    pair_R = fab["pair_avoidance"]["r"]

    # 障害物回避
    obs_avoidance_rmp = multi_robot_rmp.PairwiseObstacleAvoidance(**rmp["obstacle_avoidance"])
    obs_avoidamce_fab = fabric.ObstacleAvoidance(**fab["pair_avoidance"])
    obs_R = rmp["obstacle_avoidance"]["Ds"]

    # 目標アトラクタ
    attractor_rmp = multi_robot_rmp.UnitaryGoalAttractor_a(**rmp["goal_attractor"])
    attractor_fab = fabric.GoalAttractor(**fab["goal_attractor"])

    # 初期値選定
    if sim_param["initial_condition"]["type"] == "random":
        xu = sim_param["initial_condition"]["value"]["x_max"]
        xl = sim_param["initial_condition"]["value"]["x_min"]
        yu = sim_param["initial_condition"]["value"]["y_max"]
        yl = sim_param["initial_condition"]["value"]["y_min"]
        flag = True
        xx_s = []
        for _ in range(1000000):
            xx_s = []
            for i in range(N):
                xx_s.append((rand.uniform(xl, xu), rand.uniform(yl, yu)))
            
            # 衝突チェック
            flag = True
            for i in range(N):
                for j in range(N):
                    if i == j:
                        flag *= True
                    else:
                        if np.sqrt((xx_s[i][0]-xx_s[j][0])**2 + (xx_s[i][1]-xx_s[j][1])**2) < pair_R*1.1:
                            flag *= False
            if flag == True:
                #print("OK!")
                break
        if flag == False:
            print("error 初期値に衝突有り!")
        X0_ = []
        for i in range(N):
            X0_.extend([
                xx_s[i][0], xx_s[i][1], rand.uniform(0, 2*pi),
                0, 0, 0,
                0, 0
            ])
        X0 = np.array(X0_)
    else:
        X0_ = []
        x0 = sim_param["initial_condition"]["value"]
        for i in range(N):
            X0_.extend([
                x0[2*i], x0[2*i+1], x0[2*i+2],
                0, 0, 0,
                0, 0
            ])
        X0 = np.array(X0_)

    time_interval = sim_param["time_interval"]
    time_span = sim_param["time_span"]
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)

    L = 0.05


    def dX(t, X, sim_name):
        #print("\nt = ", t)
        
        x_s, x_dot_s = [], []
        theta_s, omega_s = [], []
        v_s = []; xi_s = []
        y_s = []  #制御点の位置．二重配列
        y_dot_s = []
        for i in range(N):
            x_s.append(np.array([X[sdim*i:sdim*i+2]]).T)
            x_dot_s.append(np.array([X[sdim*i+3:sdim*i+5]]).T)
            theta_s.append(X[sdim*i+2])
            omega_s.append(X[sdim*i+5])
            v_s.append(X[sdim*i+6])
            xi_s.append(X[sdim*i+7])

            y_, y_dot_ = calc_cpoint_state(
                x_s[-1], x_dot_s[-1], theta_s[-1], omega_s[-1], x_bar_s
            )
            y_s.append(y_)
            y_dot_s.append(y_dot_)

        X_dot = np.zeros((sdim*N, 1))  #速度ベクトル
        for i in range(N):  #ロボットごとに計算
            #print("robot_name = ", i)
            trans_M = np.zeros((2, 2))
            trans_F = np.zeros((2, 1))
            M = np.zeros((2, 2))
            F = np.zeros((2, 1))

            J, T = turtle_bot(
                x=x_s[i][0,0], y=x_s[i][1,0], theta=theta_s[i],
                dx=x_dot_s[i][0,0], dy=x_dot_s[i][1,0], omega=omega_s[i],
                v=v_s[i], xi=xi_s[i]
            )
            # J, T = car_ex(
            #     x=x_s[i][0,0], y=x_s[i][1,0], theta=theta_s[i],
            #     dx=x_dot_s[i][0,0], dy=x_dot_s[i][1,0], omega=omega_s[i],
            #     v=v_s[i], xi=xi_s[i], L=L
            # )


            if i == 0 and xg is not None:  #アトラクタ
                # head_x = x_s[i] + rotate(theta_s[i]) @ np.array([[pair_R, 0]]).T
                # head_x_dot = x_dot_s[i] + omega_s[i]*rotate_dot(theta_s[i]) @ np.array([[pair_R, 0]]).T
                head_x = y_s[i][0]
                head_x_dot = y_dot_s[i][0]
                if sim_name == "rmp":
                    M, F = attractor_rmp.calc_rmp(head_x, head_x_dot, xg)
                elif sim_name == "fabric":
                    M, F, _, _, _ = attractor_fab.calc_fabric(head_x, head_x_dot, xg)
                J_trans = J_transform(x_bar_s[0], theta_s[i])
                trans_M += (J_trans @ J).T @ M @ (J_trans @ J)
                trans_F += (J_trans @ J).T @ F

            # if xo_s is not None:
            #     for xo in xo_s:  #障害物回避
            #         if sim_name == "rmp":
            #             M, F = obs_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], xo)
            #         elif sim_name == "fabric":
            #             M, F, _, _, _, _, _ = obs_avoidamce_fab.calc_fabric(x_s[i], x_dot_s[i], xo, np.zeros(xo.shape))
            #         else:
            #             assert False
            #         root_M += M; root_F += F

            if N != 1:
                for k in range(cpoint_num): #ロボット間の回避
                    xa = y_s[i][k]
                    xa_dot = y_dot_s[i][k]
                    J_trans = J_transform(x_bar_s[k], theta_s[i])
                    for j in range(N):
                        if i != j:
                            xb = x_s[j]
                            xb_dot = x_dot_s[j]
                            if sim_name == "rmp":
                                M, F = pair_avoidance_rmp.calc_rmp(xa, xa_dot, xb)
                            elif sim_name =="fabric":
                                M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(xa, xa_dot, xb, xb_dot)
                            trans_M += (J_trans @ J).T @ M @ (J_trans @ J)
                            trans_F += (J_trans @ J).T @ F
        
        
            if len(pres_pair[i]) != 0:
                for p in pres_pair[i]:  #フォーメーション維持
                    #print("temp_pair = ", p)
                    cp_num, ib = p
                    # print("cp_num = ", cp_num)
                    # print("ib = ", ib)
                    xa = y_s[i][cp_num]
                    xa_dot = y_dot_s[i][cp_num]
                    xb = y_s[ib[0]][ib[1]]
                    # print("xa = ", xa.T)
                    # print("xb = ", xb.T)
                    # print("y_s_all =", y_s)
                    if sim_name == "rmp":
                        M, F = formation_rmp.calc_rmp(xa, xa_dot, xb)
                    elif sim_name == "fabric":
                        M, F = formation_fab.calc_rmp(xa, xa_dot, xb)
                    
                    #print("pair_F =", F.T)
                    J_trans = J_transform(x_bar_s[cp_num], theta_s[i])
                    trans_M += (J_trans @ J).T @ M @ (J_trans @ J)
                    trans_F += (J_trans @ J).T @ F


            
            # J_trans = J_transform(pair_R, 0, theta_s[i])

            # trans_M = (J_trans @ J).T @ root_M @ (J_trans @ J)
            # trans_F = (J_trans @ J).T @ (root_F)

            u_dot = LA.pinv(trans_M) @ trans_F
            
            #print("trans_F = ", trans_F.T)
            #print("du = ", u_dot.T)
            
            
            a = J @ u_dot + T

            X_dot[sdim*i+0:sdim*i+2, :] = x_dot_s[i]
            X_dot[sdim*i+2, :] = omega_s[i]
            X_dot[sdim*i+3:sdim*i+6, :] = a
            X_dot[sdim*i+6:sdim*i+8, :] = u_dot
            
            
        return np.ravel(X_dot)


    for sim_name in ["rmp", "fabric"]:
        t0 = time.perf_counter()
        sol = integrate.solve_ivp(
            fun=dX, 
            t_span=tspan, 
            y0=X0, 
            t_eval=teval, 
            args=(sim_name,)
        )
        with open(dir_base + "message/" + data_label + "_" + sim_name + '.txt', 'w') as f:
            f.write(sol.message)
            f.write("\ntime = {0}".format(time.perf_counter() - t0))

        ## CSV保存
        # まずはヘッダーを準備
        header = "t"
        for i in range(N):
            header += ",x" + str(i)
        for i in range(N):
            header += ",dx" + str(i)
        header += ",v,xi"

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

        color_list = ['b', 'g', 'm', 'c', 'y', 'r']

        # 最後の場面のグラフ
        y_s = []
        y_s_list = []
        for i in range(N):
            x = np.array([[sol.y[sdim*i][-1], sol.y[sdim*i+1][-1]]]).T
            theta = sol.y[sdim*i+2][-1]
            x_dot = np.array([[sol.y[sdim*i+3][-1], sol.y[sdim*i+4][-1]]]).T
            omega = sol.y[sdim*i+5][-1]
            y_, _ = calc_cpoint_state(
                x=x, x_dot=x_dot, theta=theta, omega=omega, x_bars=x_bar_s
            )
            y_s.append(np.concatenate(y_, axis=1))
            y_s_list.append(y_)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.plot(sol.y[sdim*i], sol.y[sdim*i+1], label="r{0}".format(i), color=color_list[i])
            c = patches.Circle(xy=(sol.y[sdim*i][-1], sol.y[sdim*i+1][-1]), radius=robot_r, ec='k', fill=False)
            ax.add_patch(c)
            
            # 制御点
            ax.scatter(y_s[i][0,1:], y_s[i][1,1:], color=color_list[i], marker=".")
            # 頭
            ax.scatter(y_s[i][0,0], y_s[i][1,0], color=color_list[i], marker="o")

        if N != 1:
            for j in range(N):
                for p in pres_pair[j]:
                    cpnum, ib = p
                    xa = y_s_list[j][cpnum]
                    xb = y_s_list[ib[0]][ib[1]]
                    frame_x = [xa[0,0], xb[0,0]]
                    frame_y = [xa[1,0], xb[1,0]]
                    ax.plot(frame_x, frame_y, color="k")

        # goal and obstacle
        if xg is not None:
            ax.scatter(
                [xg[0,0]], [xg[1,0]],
                marker="*", s=100, label="goal", color='#ff7f00',
                alpha=1, linewidths=1.5, edgecolors='red'
            )
        # for xo in xo_s:
        #     c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
        #     ax.add_patch(c)

        ax.set_title("t = {0}, and {1}".format(sol.t[-1], sol.success))
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.grid();ax.set_aspect('equal'); ax.legend()
        fig.savefig(dir_base + "fig/" + data_label + "_" + sim_name + ".png")
        plt.clf(); plt.close()




        # 状態グラフ
        fig2, axes = plt.subplots(nrows=6, ncols=1, figsize=(6, 12))
        for i in range(N):
            axes[0].plot(sol.t, sol.y[sdim*i], label="x{0}".format(i))
            axes[0].plot(sol.t, sol.y[sdim*i+1], label="y{0}".format(i))
            axes[2].plot(sol.t, sol.y[sdim*i+2], label="theta{0}".format(i))
            axes[1].plot(sol.t, sol.y[sdim*i+3], label="dx{0}".format(i))
            axes[1].plot(sol.t, sol.y[sdim*i+4], label="dy{0}".format(i))
            axes[3].plot(sol.t, sol.y[sdim*i+5], label="omega{0}".format(i))
            axes[4].plot(sol.t, sol.y[sdim*i+6], label="v{0}".format(i))
            axes[5].plot(sol.t, sol.y[sdim*i+7], label="xi{0}".format(i))
        for ax in axes.ravel():
            ax.legend()
            ax.grid()
        fig2.savefig(dir_base + "state/" + data_label + "_" + sim_name + ".png")
        plt.clf(); plt.close()


        ### アニメーション ###
        x_all, y_all = [], []
        for i in range(N):
            x_all.extend (sol.y[sdim*i])
            y_all.extend(sol.y[sdim*i+1])
        if xg is not None:
            x_all.append(xg[0,0]); y_all.append(xg[1,0])

        max_x = max(x_all)
        min_x = min(x_all)
        max_y = max(y_all)
        min_y = min(y_all)
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        max_range = max(max_x-min_x, max_y-min_y) * 0.5


        y_s = []
        y_s_list = []
        for i in range(N):
            x = np.array([[sol.y[sdim*i][0], sol.y[sdim*i+1][0]]]).T
            theta = sol.y[sdim*i+2][0]
            x_dot = np.array([[sol.y[sdim*i+3][0], sol.y[sdim*i+4][0]]]).T
            omega = sol.y[sdim*i+5][0]
            y_, _ = calc_cpoint_state(
                x=x, x_dot=x_dot, theta=theta, omega=omega, x_bars=x_bar_s
            )
            y_s.append(np.concatenate(y_, axis=1))
            y_s_list.append(y_)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        time_template = 'time = %.2f [s]'
        scale = 10
        f_scale = 0.1
        
        if xg is not None:
            ax.scatter(
                [xg[0,0]], [xg[1,0]],
                marker="*", s=100, label="goal", color='#ff7f00',
                alpha=1, linewidths=1.5, edgecolors='red'
            )
            
        # for xo in xo_s:
        #     c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
        #     ax.add_patch(c)

        tra_s = []
        robot_c_s = []
        head_s = []
        cpoint_s = []
        head_s = []
        for j in range(N):
            tra, = ax.plot(sol.y[sdim*j][:0], sol.y[sdim*j+1][:0], label="r{0}".format(j), color=color_list[j])
            tra_s.append(tra)
            c = patches.Circle(xy=(sol.y[sdim*j][0], sol.y[sdim*j+1][0]), radius=robot_r, ec='k', fill=False)
            robot_c_s.append(c)
            # xs = [sol.y[sdim*j][0] + pair_R*cos(sol.y[sdim*j+2][0])]
            # ys = [sol.y[sdim*j+1][0] + pair_R*sin(sol.y[sdim*j+2][0])]
            
            # head = ax.scatter(xs, ys)
            # head_s.append(head)
            
            # 制御点
            scat = ax.scatter(y_s[i][0,1:], y_s[i][1,1:], color=color_list[j], marker=".")
            cpoint_s.append(scat)
            
            head = ax.scatter(y_s[i][0,0], y_s[i][1,0], color=color_list[j], marker="o")
            head_s.append(head)
        
        for c in robot_c_s:
            ax.add_patch(c)

        # pair_s = {}
        # if N != 1:
        #     for j in range(N):
        #         temp_pair = []
        #         for k in pres_pair[j]:
        #             frame_x = [sol.y[sdim*k][i], sol.y[sdim*j][i]]
        #             frame_y = [sol.y[sdim*k+1][i], sol.y[sdim*j+1][i]]
        #             t_, = ax.plot(frame_x, frame_y, color="k")
        #             pair_s[(j, k)] = t_

        pair_s = {}
        if N != 1:
            for j in range(N):
                for p in pres_pair[j]:
                    cpnum, ib = p
                    xa = y_s_list[j][cpnum]
                    xb = y_s_list[ib[0]][ib[1]]
                    frame_x = [xa[0,0], xb[0,0]]
                    frame_y = [xa[1,0], xb[1,0]]
                    t_, = ax.plot(frame_x, frame_y, color="k")
                    pair_s[(j, cpnum, ib[0], ib[1])] = t_

        tx = ax.set_title(time_template % sol.t[0])
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid()
        ax.set_aspect('equal')
        ax.legend()
        
        def update(i):
            y_s = []
            y_s_list = []
            for j in range(N):
                x = np.array([[sol.y[sdim*j][i], sol.y[sdim*j+1][i]]]).T
                theta = sol.y[sdim*j+2][i]
                x_dot = np.array([[sol.y[sdim*j+3][i], sol.y[sdim*j+4][i]]]).T
                omega = sol.y[sdim*j+5][i]
                y_, _ = calc_cpoint_state(
                    x=x, x_dot=x_dot, theta=theta, omega=omega, x_bars=x_bar_s
                )
                y_s.append(np.concatenate(y_, axis=1))
                y_s_list.append(y_)
            
            
            for j in range(N):
                tra_s[j].set_data(sol.y[sdim*j][:i], sol.y[sdim*j+1][:i])
                robot_c_s[j].set_center([sol.y[sdim*j][i], sol.y[sdim*j+1][i]])
                xs = [sol.y[sdim*j][i] + pair_R*cos(sol.y[sdim*j+2][i])]
                ys = [sol.y[sdim*j+1][i] + pair_R*sin(sol.y[sdim*j+2][i])]
                #head_s[j].set_offsets([xs[0], ys[0]])
                
                d_ = []
                for k in range(1, cpoint_num):
                    d_.append((y_s_list[j][k][0,0], y_s_list[j][k][1,0]))

                cpoint_s[j].set_offsets(d_)
                head_s[j].set_offsets([(y_s_list[j][0][0,0], y_s_list[j][0][1,0])])


            # if N != 1:
            #     for j in range(N):
            #         for k in pres_pair[j]:
            #             frame_x = [sol.y[sdim*k][i], sol.y[sdim*j][i]]
            #             frame_y = [sol.y[sdim*k+1][i], sol.y[sdim*j+1][i]]
            #             pair_s[(j, k)].set_data(frame_x, frame_y)
            
            if N != 1:
                for j in range(N):
                    for p in pres_pair[j]:
                        cpnum, ib = p
                        xa = y_s_list[j][cpnum]
                        xb = y_s_list[ib[0]][ib[1]]
                        frame_x = [xa[0,0], xb[0,0]]
                        frame_y = [xa[1,0], xb[1,0]]
                        pair_s[(j, cpnum, ib[0], ib[1])].set_data(frame_x, frame_y)
            
            
            tx.set_text(time_template % sol.t[i])



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
        ani.save(dir_base + "animation/" + data_label+ sim_name + "_"  + '.gif', writer="pillow")
        plt.clf(); plt.close()

    return


def runner(sim_param, n):
    t0 = time.perf_counter()
    date_now = datetime.datetime.now()
    data_label = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    dir_base = "../syuron/formation_preservation_only/" + data_label
    os.makedirs(dir_base, exist_ok=True)
    
    itr = [
        (data_label, sim_param, i, np.random.RandomState(np.random.randint(0, 10000000)))
        for i in range(n)
    ]


    if n == 1:  # デバッグ用
        test(*itr[0])
    else:
        # 複数プロセス
        core = cpu_count()
        #core = 2
        with Pool(core) as p:
            result = p.starmap(func=test, iterable=itr)
    
    
    print("time = ", time.perf_counter() - t0)
    print("done!!")


if __name__ == "__main__":
    
    runner(car_1.sim_param, 3)
