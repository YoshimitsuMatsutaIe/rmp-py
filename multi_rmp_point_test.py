"""点モデル"""
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import time
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from scipy import integrate
from math import exp, pi, cos, sin, tan, sqrt
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

#njit
def find_random_position(xu, xl, yu, yl, n, r, rand, exists=None) -> list[list[float]]:
    """衝突の無い初期値を探す"""
    x_s = []
    new_x_s = []
    if exists is None:
        x_s.append([rand.uniform(xl, xu), rand.uniform(yl, yu)])
    else:
        x_s.extend(exists)
    
    max_trial = 10000000
    for i in range(max_trial * n):
        if len(new_x_s) == n:
            break
        else:
            tmp_x = [rand.uniform(xl, xu), rand.uniform(yl, yu)]
            flag = True
            for x in x_s:
                if len(x) == 0:
                    continue
                d = sqrt((tmp_x[0]-x[0])**2 + (tmp_x[1]-x[1])**2)
                if d < r:  #ぶつかってる
                    flag = False
                    break
            if flag:
                x_s.append(tmp_x)
                new_x_s.append(tmp_x)
    
    assert len(new_x_s) == n, "初期値生成に失敗"
    return new_x_s



def test(dir_base, sim_param, index, rand):
    """ロボット5台でテスト"""
    
    data_label = str(index)

    # xg = np.array([[4, 4]]).T
    # xo_s = [
    #     np.array([[1.0, 1.5]]).T,
    #     np.array([[2.0, 0.5]]).T,
    #     np.array([[2.5, 2.5]]).T,
    # ]
    # xg = sim_param["goal"]
    # xo_s = sim_param["obstacle"]

    robot_num = sim_param["robot_num"]
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

    ### 環境構築 ###
    robot_r = sim_param["robot_r"]
    collision_r = sim_param["collision_r"]
    goal_type = sim_param["goal"]["type"]
    obs_type = sim_param["obstacle"]["type"]
    init_p_type = sim_param["initial_condition"]["position"]["type"]
    
    goal_v = sim_param["goal"]["value"]
    obs_v = sim_param["obstacle"]["value"]
    init_p_v = sim_param["initial_condition"]["position"]["value"]
    
    _exist = []
    if goal_type == "random" and obs_type == "random" and init_p_type == "random":
        _xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"],
            yu=goal_v["y_max"], yl=goal_v["y_min"],
            n=goal_v["n"],
            r=2*collision_r,
            rand=rand
        )
        _exist.extend(_xg_s)
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"],
            yu=obs_v["y_max"], yl=obs_v["y_min"],
            n=obs_v["n"],
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
        _exist.extend(_xo_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"],
            yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            n=robot_num,
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
    
    elif goal_type != "random" and obs_type == "random" and init_p_type == "random":
        _xg_s = goal_v
        _exist.extend(_xg_s)
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"],
            yu=obs_v["y_max"], yl=obs_v["y_min"],
            n=obs_v["n"],
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
        _exist.extend(_xo_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"],
            yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            n=robot_num,
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
    
    elif goal_type == "random" and obs_type != "random" and init_p_type == "random":
        _xo_s = obs_v
        _exist.extend(_xo_s)
        _xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"],
            yu=goal_v["y_max"], yl=goal_v["y_min"],
            n=goal_v["n"],
            r=2*collision_r,
            rand=rand
        )
        _exist.extend(_xg_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"],
            yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            n=robot_num,
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
    
    elif goal_type == "random" and obs_type == "random" and init_p_type != "random":
        _x0_s = init_p_v
        _exist.extend(_x0_s)
        _xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"],
            yu=goal_v["y_max"], yl=goal_v["y_min"],
            n=goal_v["n"],
            r=2*collision_r,
            rand=rand
        )
        _exist.extend(_xg_s)
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"],
            yu=obs_v["y_max"], yl=obs_v["y_min"],
            n=obs_v["n"],
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
    
    elif goal_type != "random" and obs_type != "random" and init_p_type == "random":
        _xg_s = goal_v
        _exist.extend(_xg_s)
        _xo_s = obs_v
        _exist.extend(_xo_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"],
            yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            n=robot_num,
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )

    elif goal_type == "random" and obs_type != "random" and init_p_type != "random":
        _xo_s = obs_v
        _exist.extend(_xo_s)
        _x0_s = init_p_v
        _exist.extend(_x0_s)
        _xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"],
            yu=goal_v["y_max"], yl=goal_v["y_min"],
            n=goal_v["n"],
            r=2*collision_r,
            rand=rand
        )
    
    elif goal_type != "random" and obs_type == "random" and init_p_type != "random":
        _xg_s = goal_v
        _exist.extend(_xg_s)
        _x0_s = init_p_v
        _exist.extend(_x0_s)
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"],
            yu=obs_v["y_max"], yl=obs_v["y_min"],
            n=obs_v["n"],
            r=2*collision_r,
            rand=rand,
            exists=_exist
        )
    
    else:
        _xg_s = goal_v
        _xo_s = obs_v
        _x0_s = init_p_v

    ## ndarrayに変換 ##
    xg_s = []
    for g in _xg_s:
        if len(g) == 0:
            xg_s.append(None)
        else:
            xg_s.append(np.array([g]).T)
    
    xo_s = []
    for o in _xo_s:
        xo_s.append(np.array([o]).T)
    
    x0_s = []
    if sim_param["initial_condition"]["velocity"]["type"] == "random":
        for i in range(robot_num):
            x0_s.extend([
                _x0_s[i][0],
                _x0_s[i][1],
                rand.uniform(
                    sim_param["initial_condition"]["velocity"]["value"]["x_min"],
                    sim_param["initial_condition"]["velocity"]["value"]["x_max"]
                ),
                rand.uniform(
                    sim_param["initial_condition"]["velocity"]["value"]["y_min"],
                    sim_param["initial_condition"]["velocity"]["value"]["y_max"]
                ),
            ])
    elif sim_param["initial_condition"]["velocity"]["type"] == "zero":
        for i in range(robot_num):
            x0_s.extend([_x0_s[i][0], _x0_s[i][1], 0, 0])
    else:
        v0_s = sim_param["initial_condition"]["velocity"]["value"]
        for i in range(robot_num):
            x0_s.extend([_x0_s[i][0], _x0_s[i][1], v0_s[i][0], v0_s[i][1]])
    x0 = np.array(x0_s)



    with open("{0}/condition/initial-{1}.csv".format(dir_base, index), "w") as f:
        data = "t"
        for i in range(robot_num):
            data += ",x{0},dx{0}".format(i)
        data += "\n0.0"
        for s in x0_s:
            data += ",{0}".format(s)
        f.write(data)
    
    with open("{0}/condition/goal-{1}.csv".format(dir_base, index), "w") as f:
        data = "n,x,y\n"
        for i, g in enumerate(_xg_s):
            if len(g) == 0:
                data += "{0},NULL,NULL\n".format(i)
            else:
                data += "{0},{1},{2}\n".format(i, g[0], g[1])
        f.write(data)
    
    if len(xo_s) == 0:
        with open("{0}/condition/obs-{1}.csv".format(dir_base, index), "w") as f:
            f.write("NULL")
    else:
        _xo_con = np.concatenate(xo_s, axis=1).T
        np.savetxt(
            "{0}/condition/obs-{1}.csv".format(dir_base, index),
            _xo_con,
            header = "x,y",
            comments = '',
            delimiter = ","
        )



    rmp = sim_param["controller"]["rmp"]
    fab = sim_param["controller"]["fabric"]
    ### フォーメーション（位置）維持 ###
    distance_pres_rmp = multi_robot_rmp.ParwiseDistancePreservation_a(**rmp["formation_preservation"])
    distance_pres_fab = fabric.ParwiseDistancePreservation(**fab["formation_preservation"])
    
    
    ### フォーメーション（角度） ###
    angle_pres_fab = fabric.AnglePreservation(**fab["angle_preservation"])

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
        X_dot = np.zeros((4*robot_num, 1))
        x_s, x_dot_s = [], []
        for i in range(robot_num):
            x_s.append(np.array([[X[4*i+0], X[4*i+1]]]).T)
            x_dot_s.append(np.array([[X[4*i+2], X[4*i+3]]]).T)

        for i in range(robot_num):
            root_M = np.zeros((2, 2))
            root_F = np.zeros((2, 1))

            if xg_s[i] is not None:  #アトラクタ
                if sim_name == "rmp":
                    M, F = attractor_rmp.calc_rmp(x_s[i], x_dot_s[i], xg_s[i])
                elif sim_name == "fabric":
                    M, F, _, _, _ = attractor_fab.calc_fabric(x_s[i], x_dot_s[i], xg_s[i])
                else:
                    assert False
                #print("Fat = ", F.T)
                root_M += M; root_F += F

            for j in range(robot_num): #ロボット間の回避
                if i != j:
                    if sim_name == "rmp":
                        M, F = pair_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                    elif sim_name =="fabric":
                        M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(x_s[i], x_dot_s[i], x_s[j], x_dot_s[j])
                    else:
                        assert False
                    root_M += M; root_F += F

            if xo_s is not None:
                for xo in xo_s:  #障害物回避
                    if sim_name == "rmp":
                        M, F = obs_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], xo)
                    elif sim_name == "fabric":
                        M, F, _, _, _, _, _ = obs_avoidamce_fab.calc_fabric(x_s[i], x_dot_s[i], xo, np.zeros(xo.shape))
                    else:
                        assert False
                    root_M += M; root_F += F

            for j in pres_pair[i]:  #フォーメーション維持（距離）
                if sim_name == "rmp":
                    M, F = distance_pres_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                elif sim_name == "fabric":
                    M, F = distance_pres_fab.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                else:
                    assert False
                root_M += M; root_F += F
            
            # for j in pres_pair[i]:  #フォーメーション維持（角度）
            #     if sim_name == "rmp":
            #         M, F = distance_pres_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
            #     elif sim_name == "fabric":
            #         M, F = distance_pres_fab.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
            #     else:
            #         assert False
            #     root_M += M; root_F += F

            a = LA.pinv(root_M) @ root_F
            X_dot[4*i+0:4*i+1+1, :] = x_dot_s[i]
            X_dot[4*i+2:4*i+3+1, :] = a
            
        return np.ravel(X_dot)

    #for sim_name in ["fabric"]:
    for sim_name in ["rmp", "fabric"]:
        #t0 = time.perf_counter()
        sol = integrate.solve_ivp(
            fun=dX, 
            t_span=tspan, 
            y0=x0, 
            t_eval=teval, 
            args=(sim_name,)
        )
        #print(sol.message)
        with open("{0}/message/{1}-{2}.txt".format(dir_base, sim_name, index), 'w') as f:
            f.write(sol.message)
        #print("time = ", time.perf_counter() - t0)

        ## CSV保存
        # まずはヘッダーを準備
        header = "t"
        for i in range(robot_num):
            header += ",x{0},dx{0}".format(i)

        # 時刻歴tと解xを一つのndarrayにする
        data = np.concatenate(
            [sol.t.reshape(1, len(sol.t)).T, sol.y.T],  # sol.tは1次元配列なので2次元化する
            axis=1
        )
        np.savetxt(
            "{0}/csv/{1}-{2}.csv".format(dir_base, sim_name, index),
            data,
            header = header,
            comments = '',
            delimiter = ","
        )


        ## 状態グラフ ########################################################################
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 12))
        for i in range(robot_num):
            axes[0].plot(sol.t, sol.y[2*i], label="x{0}".format(i))
            axes[1].plot(sol.t, sol.y[2*i+1], label="y{0}".format(i))
            axes[2].plot(sol.t, sol.y[2*i+2], label="dx{0}".format(i))
            axes[3].plot(sol.t, sol.y[2*i+3], label="dy{0}".format(i))
        for ax in axes.ravel():
            ax.legend()
            ax.grid()
        fig.savefig("{0}/fig/state/{1}-{2}.jpg".format(dir_base, sim_name, index))
        plt.clf(); plt.close()


        color_list = ['b', 'g', 'm', 'c', 'y', 'r']
        ## 軌跡 ###########################################################################
        x_all, y_all = [], []
        for i in range(robot_num):
            x_all.extend (sol.y[4*i])
            y_all.extend(sol.y[4*i+1])
        
        for g in xg_s:
            if g is not None:
                x_all.append(g[0,0]); y_all.append(g[1,0])
        
        if len(xo_s) != 0:
            for o in xo_s:
                x_all.append(o[0,0]); y_all.append(o[1,0])
        
        max_x = max(x_all)
        min_x = min(x_all)
        max_y = max(y_all)
        min_y = min(y_all)
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        max_range = max(max_x-min_x, max_y-min_y) * 0.5
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(robot_num):
            ax.plot(sol.y[4*i], sol.y[4*i+1], label="r{0}".format(i), color=color_list[i])

        for j in range(robot_num):
            for k in pres_pair[j]:
                frame_x = [sol.y[4*k][-1], sol.y[4*j][-1]]
                frame_y = [sol.y[4*k+1][-1], sol.y[4*j+1][-1]]
                ax.plot(frame_x, frame_y, color="k")

        for i, g in enumerate(xg_s):
            if g is not None:
                ax.scatter([g[0,0]], [g[1,0]], marker="*", color=color_list[i], label="g{0}".format(i))
        
        if len(xo_s) != 0:
            for xo in xo_s:
                c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
                ax.add_patch(c)
            
            ax.scatter(_xo_con[:, 0], _xo_con[:, 1], marker="+", color="k", label="obs")

        ax.set_title("t = {0}, and {1}".format(sol.t[-1], sol.success))
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid();ax.set_aspect('equal'); ax.legend()
        fig.savefig("{0}/fig/trajectry/{1}-{2}.jpg".format(dir_base, sim_name, index))
        plt.clf(); plt.close()


        ## アニメ ############################################################################
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, g in enumerate(xg_s):
            if g is not None:
                ax.scatter([g[0,0]], [g[1,0]], marker="*", color=color_list[i], label="g{0}".format(i))
        
        if len(xo_s) != 0:
            for xo in xo_s:
                c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=obs_R, ec='k', fill=False)
                ax.add_patch(c)
            ax.scatter(_xo_con[:, 0], _xo_con[:, 1], marker="+", color="k", label="obs")


        robot_s = []
        traj_s = []
        for j in range(robot_num):
            c = patches.Circle(xy=(sol.y[4*j][0], sol.y[4*j+1][0]), radius=robot_r, ec='k', fill=False)
            ax.add_patch(c)
            robot_s.append(c)
            
            p, = ax.plot(sol.y[4*j][:0], sol.y[4*j+1][:0], label="r{0}".format(j), color=color_list[j])
            traj_s.append(p)

        pair_s = []
        for j in range(robot_num):
            for k in pres_pair[j]:
                frame_x = [sol.y[4*k][0], sol.y[4*j][0]]
                frame_y = [sol.y[4*k+1][0], sol.y[4*j+1][0]]
                p, = ax.plot(frame_x, frame_y, color="k")
                pair_s.append(p)


        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.grid()
        ax.set_aspect('equal')
        ax.legend()
        time_template = 'time = %.2f [s]'

        scale = 10
        f_scale = 0.1

        def update(i):
            for j in range(robot_num):
                robot_s[j].set_center([sol.y[4*j][i], sol.y[4*j+1][i]])
                traj_s[j].set_data(sol.y[4*j][:i], sol.y[4*j+1][:i])

            l = 0
            for j in range(robot_num):
                for k in pres_pair[j]:
                    frame_x = [sol.y[4*k][i], sol.y[4*j][i]]
                    frame_y = [sol.y[4*k+1][i], sol.y[4*j+1][i]]
                    pair_s[l].set_data(frame_x, frame_y)
                    l += 1

            ax.set_title(time_template % sol.t[i])

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
            
            return

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
        ani.save("{0}/fig/animation/GIF/{1}-{2}.gif".format(dir_base, sim_name, index), writer="pillow")
        ani.save("{0}/fig/animation/MP4/{1}-{2}.mp4".format(dir_base, sim_name, index), writer="ffmpeg")
        plt.clf(); plt.close()

        #print("ani_time = ", time.perf_counter() - t0)
        
    
    print("simulation {0} done!".format(index))


def runner(sim_path, n):
    date_now = datetime.datetime.now()
    today_label = date_now.strftime('%Y-%m-%d')
    os.makedirs("../syuron/point/" + today_label, exist_ok=True)
    data_label = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    dir_base = "../syuron/point/{0}/{1}".format(today_label, data_label)
    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(dir_base + "/csv", exist_ok=True)
    os.makedirs(dir_base + "/fig/trajectry", exist_ok=True)
    os.makedirs(dir_base + "/fig/animation/GIF", exist_ok=True)
    os.makedirs(dir_base + "/fig/animation/MP4", exist_ok=True)
    os.makedirs(dir_base + "/fig/state", exist_ok=True)
    os.makedirs(dir_base + "/config", exist_ok=True)
    os.makedirs(dir_base + "/message", exist_ok=True)
    os.makedirs(dir_base + "/condition", exist_ok=True)

    with open(sim_path, "r") as f:
        sim_param = yaml.safe_load(f)
    
    with open(dir_base + "/config/config.yaml", 'w') as f:
        yaml.dump(sim_param, f)
    
    
    if n == 1:
        test(dir_base, sim_param, n, np.random.RandomState(np.random.randint(0, 10000000)))
    else:
        itr = [
            (dir_base, sim_param, i, np.random.RandomState(np.random.randint(0, 10000000)))
            for i in range(n)
        ]
        core = cpu_count()
        with Pool(core) as p:
            result = p.starmap(func=test, iterable=itr)


if __name__ == "__main__":
    sim_path = "/home/matsuta_conda/src/rmp-py/config_syuron/point_1.yaml"
    runner(sim_path, 10)
