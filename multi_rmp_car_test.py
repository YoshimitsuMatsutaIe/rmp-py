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

@njit
def rotate(theta):
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

@njit
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

@njit
def J_transform(x_bar: float, y_bar: float, theta: float):
    return np.array([
        [1.0, 0.0, -sin(theta)*x_bar - cos(theta)*y_bar],
        [0.0, 1.0, cos(theta)*x_bar - sin(theta)*y_bar]
    ])


@njit
def calc_cpoint_state(x_s, x_dot_s, theta_s, omega_s, x_bars):
    y_s = []
    y_dot_s = []
    for i, x_bar in enumerate(x_bars):
        y_s.append(
            x_s[i] + rotate(theta_s[i]) @ x_bar
        )
        y_dot_s.append(
            x_dot_s[i] + omega_s[i]*rotate_dot(theta_s[i]) @ x_bar
        )
    return y_s, y_dot_s


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
    os.makedirs(dir_base + "state", exist_ok=True)
    
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
    xg = np.array([[0.5, 0.5]]).T
    #xg = None
    xo_s = None

    N = sim_param["N"]
    robot_r = sim_param["robot_r"]
    sdim = 8
    pres_pair = sim_param["pair"]
    
    cpoint_num = 4  #ロボット一台あたりの制御点の数
    x_bar_s = []
    for i in range(N):
        x_ = []
        for j in range(cpoint_num):
            theta = 2*np.pi/2 / cpoint_num * j
            x_.append(
                np.array([[robot_r*cos(theta), robot_r*sin(theta)]]).T
            )
        x_bar_s.append(x_)

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
                        if np.sqrt((xx_s[i][0]-xx_s[j][0])**2 + (xx_s[i][1]-xx_s[j][1])**2) < pair_R:
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
        #print("t = ", t)
        x_s, x_dot_s = [], []
        theta_s, omega_s = [], []
        v_s = []; xi_s = []
        for i in range(N):
            x_s.append(np.array([X[sdim*i:sdim*i+2]]).T)
            x_dot_s.append(np.array([X[sdim*i+3:sdim*i+5]]).T)
            theta_s.append(X[sdim*i+2])
            omega_s.append(X[sdim*i+5])
            v_s.append(X[sdim*i+6])
            xi_s.append(X[sdim*i+7])

        for i in range(N):
            trans_M = np.zeros((3, 3))
            trans_F = np.zeros((3, 1))
            root_M = np.zeros((2, 2))
            root_F = np.zeros((2, 1))

            if i == 0 and xg is not None:  #アトラクタ
                head_x = x_s[i] + rotate(theta_s[i]) @ np.array([[pair_R, 0]]).T
                head_x_dot = x_dot_s[i] + omega_s[i]*rotate_dot(theta_s[i]) @ np.array([[pair_R, 0]]).T
                if sim_name == "rmp":
                    M, F = attractor_rmp.calc_rmp(head_x, head_x_dot, xg)
                elif sim_name == "fabric":
                    M, F, _, _, _ = attractor_fab.calc_fabric(head_x, head_x_dot, xg)
                else:
                    assert False
                
                root_M += M
                root_F += F

            if xo_s is not None:
                for xo in xo_s:  #障害物回避
                    if sim_name == "rmp":
                        M, F = obs_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], xo)
                    elif sim_name == "fabric":
                        M, F, _, _, _, _, _ = obs_avoidamce_fab.calc_fabric(x_s[i], x_dot_s[i], xo, np.zeros(xo.shape))
                    else:
                        assert False
                    root_M += M; root_F += F

            if N != 1:
                for j in range(N): #ロボット間の回避
                    if i != j:
                        if sim_name == "rmp":
                            M, F = pair_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                        elif sim_name =="fabric":
                            M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(x_s[i], x_dot_s[i], x_s[j], x_dot_s[j])
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
            
            J_trans = J_transform(pair_R, 0, theta_s[i])

            trans_M = (J_trans @ J).T @ root_M @ (J_trans @ J)
            trans_F = (J_trans @ J).T @ (root_F)

            u_dot = LA.pinv(trans_M) @ trans_F
            #print(u_dot)
            
            
            a = J @ u_dot + T
            X_dot = np.zeros((sdim*N, 1))
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

        # 最後の場面のグラフ
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.plot(sol.y[sdim*i], sol.y[sdim*i+1], label="r{0}".format(i))
            c = patches.Circle(xy=(sol.y[sdim*i][-1], sol.y[sdim*i+1][-1]), radius=robot_r, ec='k', fill=False)
            ax.add_patch(c)
            xs = [sol.y[sdim*i][-1] + pair_R*cos(sol.y[sdim*i+2][-1])]
            ys = [sol.y[sdim*i+1][-1] + pair_R*sin(sol.y[sdim*i+2][-1])]
            ax.scatter(xs, ys)

        if N != 1:
            for j in range(N):
                for k in pres_pair[j]:
                    frame_x = [sol.y[sdim*k][-1], sol.y[sdim*j][-1]]
                    frame_y = [sol.y[sdim*k+1][-1], sol.y[sdim*j+1][-1]]
                    ax.plot(frame_x, frame_y, color="k")

        if xg is not None:
            ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
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

        fig = plt.figure()
        ax = fig.add_subplot(111)
        time_template = 'time = %.2f [s]'
        scale = 10
        f_scale = 0.1
        
        if xg is not None:
            ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
            
        # for xo in xo_s:
        #     c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
        #     ax.add_patch(c)

        tra_s = []
        robot_c_s = []
        head_s = []
        for j in range(N):
            tra, = ax.plot(sol.y[sdim*j][:0], sol.y[sdim*j+1][:0], label="r{0}".format(j))
            tra_s.append(tra)
            c = patches.Circle(xy=(sol.y[sdim*j][0], sol.y[sdim*j+1][0]), radius=pair_R, ec='k', fill=False)
            robot_c_s.append(c)
            xs = [sol.y[sdim*j][0] + pair_R*cos(sol.y[sdim*j+2][0])]
            ys = [sol.y[sdim*j+1][0] + pair_R*sin(sol.y[sdim*j+2][0])]
            
            head = ax.scatter(xs, ys)
            head_s.append(head)
        
        for c in robot_c_s:
            ax.add_patch(c)

        pair_s = {}
        if N != 1:
            for j in range(N):
                temp_pair = []
                for k in pres_pair[j]:
                    frame_x = [sol.y[sdim*k][i], sol.y[sdim*j][i]]
                    frame_y = [sol.y[sdim*k+1][i], sol.y[sdim*j+1][i]]
                    t_, = ax.plot(frame_x, frame_y, color="k")
                    pair_s[(j, k)] = t_

        tx = ax.set_title(time_template % sol.t[0])
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid()
        ax.set_aspect('equal')
        ax.legend()
        
        def update(i):
            for j in range(N):
                tra_s[j].set_data(sol.y[sdim*j][:i], sol.y[sdim*j+1][:i])
                robot_c_s[j].set_center([sol.y[sdim*j][i], sol.y[sdim*j+1][i]])
                xs = [sol.y[sdim*j][i] + pair_R*cos(sol.y[sdim*j+2][i])]
                ys = [sol.y[sdim*j+1][i] + pair_R*sin(sol.y[sdim*j+2][i])]
                head_s[j].set_offsets([xs[0], ys[0]])

            if N != 1:
                for j in range(N):
                    for k in pres_pair[j]:
                        frame_x = [sol.y[sdim*k][i], sol.y[sdim*j][i]]
                        frame_y = [sol.y[sdim*k+1][i], sol.y[sdim*j+1][i]]
                        pair_s[(j, k)].set_data(frame_x, frame_y)
            
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


def runner(sim_path, n):
    t0 = time.perf_counter()
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
        result = p.starmap(func=test, iterable=itr)
    
    print("time = ", time.perf_counter() - t0)


if __name__ == "__main__":
    sim_path = "/home/matsuta_conda/src/rmp-py/config_syuron/car_1.yaml"
    runner(sim_path, 1000)
