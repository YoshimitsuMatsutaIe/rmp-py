"""点モデル

syuron用
"""

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import time
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from scipy import integrate
from math import exp, pi, cos, sin, tan, sqrt
#from typing import Union, Tuple
from numba import njit
import datetime
import os
import yaml
import shutil


from rmp_leaf import LeafBase
import fabric
import multi_robot_rmp
import config_syuron.point_2d as p2d
import config_syuron.point_3d as p3d
from multiprocessing import Pool, cpu_count


def rotate(theta):
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])


# 五角形の計算
def pentagon(r, x, y, theta):
    r = r/2/cos(54/180*pi)*1.2
    xs = []
    for i in [0, 1, 2, 3, 4]:
        _x = rotate(theta) @ (np.array([[r*cos(2*pi/5*i + pi/2), r*sin(2*pi/5*i + pi/2)]]).T) + np.array([[x,y]]).T
        xs.append(np.ravel(_x).tolist())
    
    return xs


#njit
def find_random_position(xu, xl, yu, yl, zu, zl, n, r, rand, exists=None) -> list[list[float]]:
    """衝突の無い初期値を探す"""
    x_s = []
    new_x_s = []
    if exists is None:
        if zu is None:
            x_s.append([rand.uniform(xl, xu), rand.uniform(yl, yu)])
        else:
            x_s.append([rand.uniform(xl, xu), rand.uniform(yl, yu), rand.uniform(zl, zu)])
    else:
        x_s.extend(exists)
    
    max_trial = 10000000
    for i in range(max_trial * n):
        if len(new_x_s) == n:
            break
        else:
            if zu is None:
                tmp_x = [rand.uniform(xl, xu), rand.uniform(yl, yu)]
            else:
                tmp_x = [rand.uniform(xl, xu), rand.uniform(yl, yu), rand.uniform(zl, zu)]
            flag = True
            for x in x_s:
                #print(x)
                if len(x) == 0:
                    continue
                if zu is None:
                    d = sqrt((tmp_x[0]-x[0])**2 + (tmp_x[1]-x[1])**2)
                else:
                    d = sqrt((tmp_x[0]-x[0])**2 + (tmp_x[1]-x[1])**2 + (tmp_x[2]-x[2])**2)
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
    ROBOT_NUM = sim_param["robot_num"]
    pres_pair = sim_param["pair"]
    #pres_angle_pair = sim_param["angle_pair"]

    ### 環境構築 ###
    TASK_DIM = sim_param["task_dim"]
    ROBOT_R = sim_param["robot_r"]
    COLLISION_R = sim_param["collision_r"]
    goal_type = sim_param["goal"]["type"]
    obs_type = sim_param["obstacle"]["type"]
    init_p_type = sim_param["initial_condition"]["position"]["type"]
    
    goal_v = sim_param["goal"]["value"]
    obs_v = sim_param["obstacle"]["value"]
    init_p_v = sim_param["initial_condition"]["position"]["value"]
    
    _exist = []
    if goal_type == "random" and obs_type == "random" and init_p_type == "random":
        _g_point = goal_v["point"]
        _g_num = 0
        for _g in _g_point:
            if _g:
                _g_num += 1
        
        __xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"], yu=goal_v["y_max"], yl=goal_v["y_min"],
            zu=goal_v["z_max"] if TASK_DIM==3 else None,
            zl=goal_v["z_min"] if TASK_DIM==3 else None,
            n=_g_num, r=2*COLLISION_R, rand=rand
        )
        _exist.extend(__xg_s)
        _xg_s = []
        for _g in _g_point:
            if _g:
                _xg_s.append(__xg_s.pop())
            else:
                _xg_s.append([])
        
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"], yu=obs_v["y_max"], yl=obs_v["y_min"],
            zu=obs_v["z_max"] if TASK_DIM==3 else None,
            zl=obs_v["z_min"] if TASK_DIM==3 else None,
            n=obs_v["n"], r=2*COLLISION_R, rand=rand, exists=_exist
        )
        _exist.extend(_xo_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"], yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            zu=init_p_v["z_max"] if TASK_DIM==3 else None,
            zl=init_p_v["z_min"] if TASK_DIM==3 else None,
            n=ROBOT_NUM, r=2*COLLISION_R, rand=rand, exists=_exist
        )
    
    elif goal_type != "random" and obs_type == "random" and init_p_type == "random":
        _xg_s = goal_v
        _exist.extend(_xg_s)
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"], yu=obs_v["y_max"], yl=obs_v["y_min"],
            zu=obs_v["z_max"] if TASK_DIM==3 else None,
            zl=obs_v["z_min"] if TASK_DIM==3 else None,
            n=obs_v["n"], r=2*COLLISION_R, rand=rand, exists=_exist
        )
        _exist.extend(_xo_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"], yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            zu=init_p_v["z_max"] if TASK_DIM==3 else None,
            zl=init_p_v["z_min"] if TASK_DIM==3 else None,
            n=ROBOT_NUM, r=2*COLLISION_R,
            rand=rand, exists=_exist
        )
    
    elif goal_type == "random" and obs_type != "random" and init_p_type == "random":
        _xo_s = obs_v
        _exist.extend(_xo_s)
        
        _g_point = goal_v["point"]
        _g_num = 0
        for _g in _g_point:
            if _g:
                _g_num += 1
        
        __xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"], yu=goal_v["y_max"], yl=goal_v["y_min"],
            zu=goal_v["z_max"] if TASK_DIM==3 else None,
            zl=goal_v["z_min"] if TASK_DIM==3 else None,
            n=_g_num, r=2*COLLISION_R, rand=rand
        )
        _exist.extend(__xg_s)
        _xg_s = []
        for _g in _g_point:
            if _g:
                _xg_s.append(__xg_s.pop())
            else:
                _xg_s.append([])
        
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"], yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            zu=init_p_v["z_max"] if TASK_DIM==3 else None,
            zl=init_p_v["z_min"] if TASK_DIM==3 else None,
            n=ROBOT_NUM, r=2*COLLISION_R, rand=rand, exists=_exist
        )
    
    elif goal_type == "random" and obs_type == "random" and init_p_type != "random":
        _x0_s = init_p_v
        _exist.extend(_x0_s)
        
        _g_point = goal_v["point"]
        _g_num = 0
        for _g in _g_point:
            if _g:
                _g_num += 1
        
        __xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"], yu=goal_v["y_max"], yl=goal_v["y_min"],
            zu=goal_v["z_max"] if TASK_DIM==3 else None,
            zl=goal_v["z_min"] if TASK_DIM==3 else None,
            n=_g_num, r=2*COLLISION_R, rand=rand
        )
        _exist.extend(__xg_s)
        _xg_s = []
        for _g in _g_point:
            if _g:
                _xg_s.append(__xg_s.pop())
            else:
                _xg_s.append([])
        
        
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"], yu=obs_v["y_max"], yl=obs_v["y_min"],
            zu=obs_v["z_max"] if TASK_DIM==3 else None,
            zl=obs_v["z_min"] if TASK_DIM==3 else None,
            n=obs_v["n"], r=2*COLLISION_R, rand=rand, exists=_exist
        )
    
    elif goal_type != "random" and obs_type != "random" and init_p_type == "random":
        _xg_s = goal_v
        _exist.extend(_xg_s)
        _xo_s = obs_v
        _exist.extend(_xo_s)
        _x0_s = find_random_position(
            xu=init_p_v["x_max"], xl=init_p_v["x_min"], yu=init_p_v["y_max"], yl=init_p_v["y_min"],
            zu=init_p_v["z_max"] if TASK_DIM==3 else None,
            zl=init_p_v["z_min"] if TASK_DIM==3 else None,
            n=ROBOT_NUM, r=2*COLLISION_R, rand=rand, exists=_exist
        )

    elif goal_type == "random" and obs_type != "random" and init_p_type != "random":
        _xo_s = obs_v
        _exist.extend(_xo_s)
        _x0_s = init_p_v
        _exist.extend(_x0_s)
    
        _g_point = goal_v["point"]
        _g_num = 0
        for _g in _g_point:
            if _g:
                _g_num += 1
        
        __xg_s = find_random_position(
            xu=goal_v["x_max"], xl=goal_v["x_min"], yu=goal_v["y_max"], yl=goal_v["y_min"],
            zu=goal_v["z_max"] if TASK_DIM==3 else None,
            zl=goal_v["z_min"] if TASK_DIM==3 else None,
            n=_g_num, r=2*COLLISION_R, rand=rand
        )
        _exist.extend(__xg_s)
        _xg_s = []
        for _g in _g_point:
            if _g:
                _xg_s.append(__xg_s.pop())
            else:
                _xg_s.append([])
    
    
    elif goal_type != "random" and obs_type == "random" and init_p_type != "random":
        _xg_s = goal_v
        _exist.extend(_xg_s)
        _x0_s = init_p_v
        _exist.extend(_x0_s)
        _xo_s = find_random_position(
            xu=obs_v["x_max"], xl=obs_v["x_min"], yu=obs_v["y_max"], yl=obs_v["y_min"],
            zu=obs_v["z_max"] if TASK_DIM==3 else None,
            zl=obs_v["z_min"] if TASK_DIM==3 else None,
            n=obs_v["n"], r=2*COLLISION_R, rand=rand, exists=_exist
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
        for i in range(ROBOT_NUM):
            if TASK_DIM == 2:
                x0_s.extend([
                    _x0_s[i][0], _x0_s[i][1],
                    rand.uniform(
                        sim_param["initial_condition"]["velocity"]["value"]["x_min"],
                        sim_param["initial_condition"]["velocity"]["value"]["x_max"]
                    ),
                    rand.uniform(
                        sim_param["initial_condition"]["velocity"]["value"]["y_min"],
                        sim_param["initial_condition"]["velocity"]["value"]["y_max"]
                    ),
                ])
            else:
                x0_s.extend([
                    _x0_s[i][0], _x0_s[i][1], _x0_s[i][2],
                    rand.uniform(
                        sim_param["initial_condition"]["velocity"]["value"]["x_min"],
                        sim_param["initial_condition"]["velocity"]["value"]["x_max"]
                    ),
                    rand.uniform(
                        sim_param["initial_condition"]["velocity"]["value"]["y_min"],
                        sim_param["initial_condition"]["velocity"]["value"]["y_max"]
                    ),
                    rand.uniform(
                        sim_param["initial_condition"]["velocity"]["value"]["z_min"],
                        sim_param["initial_condition"]["velocity"]["value"]["z_max"]
                    ),
                ])
    elif sim_param["initial_condition"]["velocity"]["type"] == "zero":
        for i in range(ROBOT_NUM):
            if TASK_DIM == 2:
                x0_s.extend([_x0_s[i][0], _x0_s[i][1], 0, 0])
            else:
                x0_s.extend([_x0_s[i][0], _x0_s[i][1], _x0_s[i][2], 0, 0, 0])
    else:
        v0_s = sim_param["initial_condition"]["velocity"]["value"]
        for i in range(ROBOT_NUM):
            if TASK_DIM == 2:
                x0_s.extend([_x0_s[i][0], _x0_s[i][1], v0_s[i][0], v0_s[i][1]])
            else:
                x0_s.extend([_x0_s[i][0], _x0_s[i][1], _x0_s[i][2], v0_s[i][0], v0_s[i][1], v0_s[i][2]])
    x0 = np.array(x0_s)



    with open("{0}/condition/initial-{1}.csv".format(dir_base, index), "w") as f:
        data = "t"
        for i in range(ROBOT_NUM):
            if TASK_DIM == 2:
                data += ",x{0},y{0},dx{0},dy{0}".format(i)
            else:
                data += ",x{0},y{0},z{0},dx{0},dy{0},dz{0}".format(i)
        data += "\n0.0"
        for s in x0_s:
            data += ",{0}".format(s)
        f.write(data)
    
    with open("{0}/condition/goal-{1}.csv".format(dir_base, index), "w") as f:
        data = "n,x,y\n" if TASK_DIM==2 else "n,x,y,z\n"
        for i, g in enumerate(_xg_s):
            if len(g) == 0:
                if TASK_DIM == 2:
                    data += "{0},NULL,NULL\n".format(i)
                else:
                    data += "{0},NULL,NULL,NULL\n".format(i)
            else:
                if TASK_DIM == 2:
                    data += "{0},{1},{2}\n".format(i, g[0], g[1])
                else:
                    data += "{0},{1},{2},{3}\n".format(i, g[0], g[1], g[2])
        f.write(data)
    
    if len(xo_s) == 0:
        with open("{0}/condition/obs-{1}.csv".format(dir_base, index), "w") as f:
            f.write("NULL")
    else:
        _xo_con = np.concatenate(xo_s, axis=1).T
        np.savetxt(
            "{0}/condition/obs-{1}.csv".format(dir_base, index),
            _xo_con,
            header = "x,y" if TASK_DIM == 2 else "x,y,z",
            comments = '',
            delimiter = ","
        )



    rmp = sim_param["controller"]["rmp"]
    fab = sim_param["controller"]["fabric"]
    ### フォーメーション（位置）維持 ###
    distance_pres_rmp = multi_robot_rmp.ParwiseDistancePreservation_a(**rmp["formation_preservation"])
    distance_pres_fab = fabric.ParwiseDistancePreservation(**fab["formation_preservation"])
    FORMATION_PRESERVARION_R = sim_param["formation_preservation_r"]
    
    # ### フォーメーション（角度） ###
    # angle_pres_fab = fabric.AnglePreservation(**fab["angle_preservation"])

    # ロボット間の障害物回避
    pair_avoidance_rmp = multi_robot_rmp.PairwiseObstacleAvoidance(**rmp["pair_avoidance"])
    pair_avoidance_fab = fabric.ObstacleAvoidance2(**fab["pair_avoidance"])

    # 障害物回避
    obs_avoidance_rmp = multi_robot_rmp.PairwiseObstacleAvoidance(**rmp["obstacle_avoidance"])
    obs_avoidamce_fab = fabric.ObstacleAvoidance2(**fab["obstacle_avoidance"])
    #obs_R = rmp["obstacle_avoidance"]["Ds"]

    # 目標アトラクタ
    attractor_rmp = multi_robot_rmp.UnitaryGoalAttractor_a(**rmp["goal_attractor"])
    attractor_fab = fabric.GoalAttractor(**fab["goal_attractor"])

    
    # 移動空間制限
    
    #limit_avoidance_fab = fabric.SpaceLimitAvoidance(**fab["space_limit_avoidance"])
    limit_avoidance_fab = fabric.SpaceLimitAvoidance_2(**fab["space_limit_avoidance"])
    
    
    time_interval = sim_param["time_interval"]
    time_span = sim_param["time_span"]
    tspan = (0, time_span)
    teval = np.arange(0, time_span, time_interval)

    # v_ = -0.01
    # o0_ = np.array([[0.4, 0.0]]).T
    # xo_s = [
    #         o0_ + np.array([[v_*0, 0.]]).T
    #     ]
    # _xo_con = np.concatenate(xo_s, axis=1).T
    
    def dX_rmp(t, X, sim_name):
        #print("t = ", t)
        X_dot = np.zeros((2*TASK_DIM*ROBOT_NUM, 1))
        x_s, x_dot_s = [], []
        for i in range(ROBOT_NUM):
            x_s.append(np.array([X[2*TASK_DIM*i:2*TASK_DIM*i+TASK_DIM]]).T)
            x_dot_s.append(np.array([X[2*TASK_DIM*i+TASK_DIM:2*TASK_DIM*i+2*TASK_DIM]]).T)

        root_M_all = np.zeros((ROBOT_NUM*TASK_DIM, ROBOT_NUM*TASK_DIM))
        root_F_all = np.zeros((ROBOT_NUM*TASK_DIM, 1))
        
        for i in range(ROBOT_NUM):
            #print("i = ", i)
            root_M = np.zeros((TASK_DIM, TASK_DIM))
            root_F = np.zeros((TASK_DIM, 1))
            M = np.zeros((TASK_DIM, TASK_DIM))
            F = np.zeros((TASK_DIM, 1))

            if xg_s[i] is not None:  #アトラクタ
                M, F = attractor_rmp.calc_rmp(x_s[i], x_dot_s[i], xg_s[i])
                #print("Fat = ", F.T)
                root_M += M; root_F += F

            for j in range(ROBOT_NUM): #ロボット間の回避
                if i != j:
                    M, F = pair_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], x_s[j])
                    root_M += M; root_F += F

            if xo_s is not None:
                for xo in xo_s:  #障害物回避
                    M, F = obs_avoidance_rmp.calc_rmp(x_s[i], x_dot_s[i], xo)
                    root_M += M; root_F += F

            for p in pres_pair[i]:  #フォーメーション維持（距離）
                j, d = p
                M, F = distance_pres_rmp.calc_rmp(d, x_s[i], x_dot_s[i], x_s[j])
                root_M += M; root_F += F
            
            root_M_all[TASK_DIM*i:TASK_DIM*(i+1), TASK_DIM*i:TASK_DIM*(i+1)] = root_M
            root_F_all[TASK_DIM*i:TASK_DIM*(i+1), :] = root_F

            X_dot[2*TASK_DIM*i+0:2*TASK_DIM*i+TASK_DIM, :] = x_dot_s[i]
        
        a_all = LA.pinv(root_M_all) @ root_F_all
        for i in range(ROBOT_NUM):
            X_dot[2*TASK_DIM*i+TASK_DIM:2*TASK_DIM*i+2*TASK_DIM, :] = a_all[TASK_DIM*i:TASK_DIM*(i+1), :]
        
        return np.ravel(X_dot)

    
    def dX_fab(t, X, sim_name):
        #print("t = ", t)
        X_dot = np.zeros((2*TASK_DIM*ROBOT_NUM, 1))
        x_s, x_dot_s = [], []
        for i in range(ROBOT_NUM):
            x_s.append(np.array([X[2*TASK_DIM*i:2*TASK_DIM*i+TASK_DIM]]).T)
            x_dot_s.append(np.array([X[2*TASK_DIM*i+TASK_DIM:2*TASK_DIM*i+2*TASK_DIM]]).T)



        root_M_all = np.zeros((ROBOT_NUM*TASK_DIM, ROBOT_NUM*TASK_DIM))
        root_F_all = np.zeros((ROBOT_NUM*TASK_DIM, 1))
        
        for i in range(ROBOT_NUM):
            #print("i = ", i)
            root_M = np.zeros((TASK_DIM, TASK_DIM)); root_F = np.zeros((TASK_DIM, 1))
            M = np.zeros((TASK_DIM, TASK_DIM)); F = np.zeros((TASK_DIM, 1))

            if xg_s[i] is not None:  #アトラクタ
                M, F, _, _, _ = attractor_fab.calc_fabric(x_s[i], x_dot_s[i], xg_s[i])
                #print("Fat = ", F.T)
                root_M += M; root_F += F

            for j in range(ROBOT_NUM): #ロボット間の回避
                if i != j:
                    M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(
                        x_s[i], x_dot_s[i], x_s[j], x_dot_s[j]
                    )
                    root_M += M; root_F += F

            if xo_s is not None:
                for xo in xo_s:  #障害物回避
                    M, F, _, _, _, _, _ = obs_avoidamce_fab.calc_fabric(
                        x_s[i], x_dot_s[i], xo, np.zeros(xo.shape)
                    )
                    #print(F)
                    root_M += M; root_F += F

            for p in pres_pair[i]:  #フォーメーション維持（距離）
                #print(j)
                j, d = p
                M, F = distance_pres_fab.calc_rmp(d, x_s[i], x_dot_s[i], x_s[j])
                root_M += M; root_F += F
            
            root_M_all[TASK_DIM*i:TASK_DIM*(i+1), TASK_DIM*i:TASK_DIM*(i+1)] = root_M
            root_F_all[TASK_DIM*i:TASK_DIM*(i+1), :] = root_F
            X_dot[2*TASK_DIM*i+0:2*TASK_DIM*i+TASK_DIM, :] = x_dot_s[i]
        
        a_all = LA.pinv(root_M_all) @ root_F_all
        for i in range(ROBOT_NUM):
            X_dot[2*TASK_DIM*i+TASK_DIM:2*TASK_DIM*i+2*TASK_DIM, :] = a_all[TASK_DIM*i:TASK_DIM*(i+1), :]
        
        return np.ravel(X_dot)


    
    def dX_fab_pair(t, X, sim_name):
        """リーダーは0のみ．他はリーダーに追従"""
        #print(X.shape)
        #print("t = ", t)
        ROBOT_NUM2 = 2*ROBOT_NUM - 1
        X_dot = np.zeros((2*TASK_DIM*(ROBOT_NUM2), 1))
        x_s, x_dot_s = [], []
        for i in range(ROBOT_NUM2):
            x_s.append(np.array([X[2*TASK_DIM*i:2*TASK_DIM*i+TASK_DIM]]).T)
            x_dot_s.append(np.array([X[2*TASK_DIM*i+TASK_DIM:2*TASK_DIM*i+2*TASK_DIM]]).T)

        root_M_all = np.zeros((ROBOT_NUM2*TASK_DIM, ROBOT_NUM2*TASK_DIM))
        root_F_all = np.zeros((ROBOT_NUM2*TASK_DIM, 1))
        


        
        for i in range(ROBOT_NUM2):
            #print("i = ", i)
            root_M = np.zeros((TASK_DIM, TASK_DIM)); root_F = np.zeros((TASK_DIM, 1))
            M = np.zeros((TASK_DIM, TASK_DIM)); F = np.zeros((TASK_DIM, 1))

            if i == 0:  # リーダー
                M, F, _, _, _ = attractor_fab.calc_fabric(x_s[0], x_dot_s[0], xg_s[0])
                #print("Fat = ", F.T)
                root_M += M; root_F += F
            elif i < ROBOT_NUM:  # 仮想目標へフォロワロボットが追従
                
                M, F, _, _, _ = attractor_fab.calc_fabric(x_s[i], x_dot_s[i], x_s[i+ROBOT_NUM-1])
                #print("Fat = ", F.T)
                root_M += M; root_F += F
            else:
                pass
            
            
            if i < ROBOT_NUM: #リアルロボット
                for j in range(ROBOT_NUM): #ロボット間の回避
                    if i != j:
                        M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(
                            x_s[i], x_dot_s[i], x_s[j], x_dot_s[j]
                        )
                        root_M += M; root_F += F
            else:
                for j in range(ROBOT_NUM, ROBOT_NUM2): #ロボット間の回避
                    if i != j:
                        M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(
                            x_s[i], x_dot_s[i], x_s[0], x_dot_s[0]
                        )
                        root_M += M; root_F += F
                        M, F, _, _, _, _, _ = pair_avoidance_fab.calc_fabric(
                            x_s[i], x_dot_s[i], x_s[j], x_dot_s[j]
                        )
                        root_M += M; root_F += F

            if xo_s is not None:
                for xo in xo_s:  #障害物回避
                    M, F, _, _, _, _, _ = obs_avoidamce_fab.calc_fabric(
                        x_s[i], x_dot_s[i], xo, np.zeros(xo.shape)
                    )
                    #print(F)
                    root_M += M; root_F += F

            if i == 0:
                for p in pres_pair[0]:  #フォーメーション維持（距離）
                    j, d = p
                    #print(j)
                    M, F = distance_pres_fab.calc_rmp(d, x_s[i], x_dot_s[i], x_s[j+ROBOT_NUM-1])
                    root_M += M; root_F += F
                pass
            elif i < ROBOT_NUM:
                pass
            else:
                for p in pres_pair[i-(ROBOT_NUM-1)]:  #フォーメーション維持（距離）
                    #print(j)
                    j, d = p
                    M, F = distance_pres_fab.calc_rmp(d, x_s[i], x_dot_s[i], x_s[j])
                    root_M += M; root_F += F
            
            
            # # 移動空間制限
            # M, F = limit_avoidance_fab.calc_rmp(x_s[i], x_dot_s[i])
            # root_M += M; root_F += F
            
            
            root_M_all[TASK_DIM*i:TASK_DIM*(i+1), TASK_DIM*i:TASK_DIM*(i+1)] = root_M
            root_F_all[TASK_DIM*i:TASK_DIM*(i+1), :] = root_F
            X_dot[2*TASK_DIM*i+0:2*TASK_DIM*i+TASK_DIM, :] = x_dot_s[i]
        
        a_all = LA.pinv(root_M_all) @ root_F_all
        for i in range(ROBOT_NUM2):
            X_dot[2*TASK_DIM*i+TASK_DIM:2*TASK_DIM*i+2*TASK_DIM, :] = a_all[TASK_DIM*i:TASK_DIM*(i+1), :]
        
        return np.ravel(X_dot)



    #for sim_name in ["fabric"]:
    for sim_name in ["rmp", "fabric"]:
        t0 = time.perf_counter()
        if sim_name == "rmp":
            sol = integrate.solve_ivp(
                fun=dX_rmp, 
                t_span=tspan, 
                y0=x0, 
                t_eval=teval, 
                args=(sim_name,)
            )
        elif sim_name == "fabric":
            
            # 5角形のとき
            g_ = xg_s[0]
            a_ =  g_[0,0] - x0[0]
            b_ =  g_[1,0] - x0[1]
            theta = np.arctan2(b_, a_) - pi/2
            fc_x = x0[0] - FORMATION_PRESERVARION_R*cos(theta+ pi/2)
            fc_y = x0[1] - FORMATION_PRESERVARION_R*sin(theta + pi/2)
            _pentagon = pentagon(FORMATION_PRESERVARION_R, 0, 0, theta)
            _x0_fab = []
            for _x in _x0_s:
                _x0_fab.extend([_x[0], _x[1], 0, 0])
            for _x in _pentagon[1:]:
                _x0_fab.extend([_x[0]+fc_x, _x[1]+fc_y, 0, 0])
            x0_fab = np.array(_x0_fab)
            
            sol = integrate.solve_ivp(
                fun=dX_fab_pair, 
                t_span=tspan, 
                y0=x0_fab, 
                t_eval=teval, 
                args=(sim_name,)
            )
        else:
            assert False
        calc_time = time.perf_counter() - t0
        #print(sol.message)
        print("{0}-{1}, time = {2}".format(sim_name, index, calc_time))
        with open("{0}/message/{1}/{2}.csv".format(dir_base, sim_name, index), 'w') as f:
            f.write("{0},{1}".format(index, sol.message))
        with open("{0}/time/{1}/{2}.csv".format(dir_base, sim_name, index), 'w') as f:
            f.write("{0},{1}".format(index, calc_time))
        
        
        
        # スコアを記録
        if sol.success:
            x_s_last = []
            for i in range(ROBOT_NUM):
                x_ = []
                for j in range(TASK_DIM):
                    x_.append(sol.y[TASK_DIM*(i+j)][-1])
                x_s_last.append(np.array([x_]).T)
        
            eg = LA.norm(xg_s[0] - x_s_last[0])
            ef = 0
            count_ = 0
            for i in range(ROBOT_NUM):
                for p in pres_pair[i]:
                    j, d = p
                    ef += (abs(d - LA.norm(x_s_last[i] - x_s_last[j])))# / d
                    count_ += 1
            ef /= count_
            
            with open("{0}/score/{1}/{2}.csv".format(dir_base, sim_name, index), 'w') as f:
                f.write("{0},{1},{2}".format(index, eg, ef))
        else:
            # with open("{0}/score/{1}/{2}-{3}.csv".format(dir_base, sim_name, sim_name, index), 'w') as f:
            #     f.write("{0},NULL,NULL".format(index))
            pass

        ## CSV保存
        # まずはヘッダーを準備
        header = "t"
        for i in range(ROBOT_NUM):
            if TASK_DIM == 2:
                header += ",x{0},y{0},dx{0},dy{0}".format(i)
            else:
                header += ",x{0},y{0},z{0},dx{0},dy{0},dz{0}".format(i)
        if len(sol.y) != 2*TASK_DIM*ROBOT_NUM:
            for i in range(ROBOT_NUM, 2*ROBOT_NUM-1):
                if TASK_DIM == 2:
                    header += ",vx{0},vy{0},dvx{0},dvy{0}".format(i-(ROBOT_NUM-1))
                else:
                    header += ",vx{0},vy{0},vz{0},dvx{0},dvy{0},dvz{0}".format(i-(ROBOT_NUM-1))

        # 時刻歴tと解xを一つのndarrayにする
        data = np.concatenate(
            [sol.t.reshape(1, len(sol.t)).T, sol.y.T],  # sol_tは1次元配列なので2次元化する
            axis=1
        )
        np.savetxt(
            "{0}/csv/{1}/{2}.csv".format(dir_base, sim_name, index),
            data,
            header = header,
            comments = '',
            delimiter = ","
        )


        sol_t = sol.t
        sol_y = [sol.y[i] for i in range(len(sol.y))]
        


        ## 状態グラフ ########################################################################
        
        
        
        
        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(6, 12)
        )
        for i in range(ROBOT_NUM):
            axes[0].plot(sol_t, sol_y[2*i], label="x{0}".format(i))
            axes[1].plot(sol_t, sol_y[2*i+1], label="y{0}".format(i))
            axes[2].plot(sol_t, sol_y[2*i+2], label="dx{0}".format(i))
            axes[3].plot(sol_t, sol_y[2*i+3], label="dy{0}".format(i))
        
        if len(sol_y) != 2*TASK_DIM*ROBOT_NUM:  # 仮想ロボットあり
            for i in range(ROBOT_NUM, 2*ROBOT_NUM-1):
                axes[0].plot(sol_t, sol_y[2*i], label="xg{0}".format(i-(ROBOT_NUM-1)))
                axes[1].plot(sol_t, sol_y[2*i+1], label="yg{0}".format(i-(ROBOT_NUM-1)))
                axes[2].plot(sol_t, sol_y[2*i+2], label="dxg{0}".format(i-(ROBOT_NUM-1)))
                axes[3].plot(sol_t, sol_y[2*i+3], label="dyg{0}".format(i-(ROBOT_NUM-1)))
            
        for ax in axes.ravel():
            ax.legend()
            ax.grid()
        fig.savefig("{0}/fig/state/{1}/{2}.jpg".format(dir_base, sim_name, index))
        plt.clf(); plt.close()


        color_list = ['b', 'g', 'm', 'c', 'y', 'r']
        ## 軌跡 ###########################################################################
        x_all, y_all, z_all = [], [], []
        for i in range(ROBOT_NUM):
            x_all.extend(sol_y[2*TASK_DIM*i])
            y_all.extend(sol_y[2*TASK_DIM*i+1])
            if TASK_DIM == 3:
                z_all.extend(sol_y[2*TASK_DIM*i+2])
        
        if len(sol_y) != 2*TASK_DIM*ROBOT_NUM:  # 仮想ロボットあり
            for i in range(ROBOT_NUM, 2*ROBOT_NUM-1):
                x_all.extend(sol_y[2*TASK_DIM*i])
                y_all.extend(sol_y[2*TASK_DIM*i+1])
                if TASK_DIM == 3:
                    z_all.extend(sol_y[2*TASK_DIM*i+2])
        
        for g in xg_s:
            if g is not None:
                x_all.append(g[0,0]); y_all.append(g[1,0])
                if TASK_DIM == 3:
                    z_all.append(g[2,0])
        
        if len(xo_s) != 0:
            for o in xo_s:
                x_all.append(o[0,0]); y_all.append(o[1,0])
                if TASK_DIM == 3:
                    z_all.append(o[2,0])
        
        max_x = max(x_all); min_x = min(x_all)
        max_y = max(y_all); min_y = min(y_all)
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        if TASK_DIM == 2:
            max_range = max(max_x-min_x, max_y-min_y) * 0.5
        else:
            max_z = max(z_all); min_z = min(z_all)
            mid_z = (max_z + min_z) * 0.5
            max_range = max(max_x-min_x, max_y-min_y, max_z-min_z) * 0.5
    
    
        epoch_max = 80
        if len(sol_t) < epoch_max:
            step = 1
        else:
            step = len(sol_t) // epoch_max

        ## グラフ ##########################################################################
        if TASK_DIM == 2:  ## 2次元 ############################################################
            fig = plt.figure() ## 最後の静止画 ##########################################################
            ax = fig.add_subplot(111)
            for i in range(ROBOT_NUM):
                ax.plot(sol_y[4*i], sol_y[4*i+1], label="r{0}".format(i), color=color_list[i])
                ax.plot(
                    sol_y[4*i][-1], sol_y[4*i+1][-1],
                    label="r{0}".format(i), marker="o", color=color_list[i]
                )

            if len(sol_y) != 2*TASK_DIM*ROBOT_NUM:  # 仮想ロボットあり
                for i in range(ROBOT_NUM, 2*ROBOT_NUM-1):
                    ax.plot(
                        sol_y[4*i], sol_y[4*i+1],
                        label="v{0}".format(i+1), color=color_list[i-(ROBOT_NUM-1)],
                        linestyle="dashed"
                    )
                    ax.plot(
                        sol_y[4*i][-1], sol_y[4*i+1][-1],
                        label="r{0}".format(i-(ROBOT_NUM-1)), marker="^", 
                        color=color_list[i-(ROBOT_NUM-1)]
                    )


            for j in range(ROBOT_NUM):
                for p in pres_pair[j]:
                    k, _ = p
                    frame_x = [sol_y[4*k][-1], sol_y[4*j][-1]]
                    frame_y = [sol_y[4*k+1][-1], sol_y[4*j+1][-1]]
                    ax.plot(frame_x, frame_y, color="k")

            for i, g in enumerate(xg_s):
                if g is not None:
                    ax.scatter([g[0,0]], [g[1,0]], marker="*", color=color_list[i], label="g{0}".format(i))
            
            if len(xo_s) != 0:
                for xo in xo_s:
                    c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=COLLISION_R, ec='k', fill=False)
                    ax.add_patch(c)
                ax.scatter(_xo_con[:, 0], _xo_con[:, 1], marker="+", color="k", label="obs")

            ax.set_title("t = {0}, and {1}".format(sol_t[-1], sol.success))
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
            ax.set_xlim(mid_x-max_range, mid_x+max_range)
            ax.set_ylim(mid_y-max_range, mid_y+max_range)
            ax.grid();ax.set_aspect('equal')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            fig.savefig("{0}/fig/trajectry/{1}/{2}.jpg".format(dir_base, sim_name, index))
            plt.clf(); plt.close()


            ## アニメ ############################################################################
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for i, g in enumerate(xg_s):
                if g is not None:
                    ax.scatter([g[0,0]], [g[1,0]], marker="*", color=color_list[i], label="g{0}".format(i))
            
            if len(xo_s) != 0:
                o_s = ax.scatter(_xo_con[:, 0], _xo_con[:, 1], marker="+", color="k", label="obs")
                o_c_s = []
                for xo in xo_s:
                    c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=COLLISION_R, ec='k', fill=False)
                    o_c_s.append(c)
                    ax.add_patch(c)
                


            robot_s = []
            traj_s = []
            center_s = []
            for j in range(ROBOT_NUM):
                c = patches.Circle(xy=(sol_y[4*j][0], sol_y[4*j+1][0]), radius=ROBOT_R, ec='k', fill=False)
                ax.add_patch(c)
                robot_s.append(c)
                
                p, = ax.plot(sol_y[4*j][:0], sol_y[4*j+1][:0], label="r{0}".format(j), color=color_list[j])
                traj_s.append(p)
                
                p, = ax.plot(
                    sol_y[4*j][0], sol_y[4*j+1][0],
                    label="r{0}".format(j), marker="o", color=color_list[j]
                )
                center_s.append(p)

            if len(sol_y) != 2*TASK_DIM*ROBOT_NUM:  # 仮想ロボットあり
                v_robot_s = []
                v_traj_s = []
                v_center_s = []
                for j in range(ROBOT_NUM, 2*ROBOT_NUM-1):
                    c = patches.Circle(xy=(sol_y[4*j][0], sol_y[4*j+1][0]), radius=ROBOT_R, ec='r', fill=False, linestyle="dashed")
                    ax.add_patch(c)
                    v_robot_s.append(c)
                    p, = ax.plot(
                        sol_y[4*j][:0], sol_y[4*j+1][:0],
                        label="v{0}".format(j+1), color=color_list[j-(ROBOT_NUM-1)],
                        linestyle="dashed"
                    )
                    v_traj_s.append(p)

                    p, = ax.plot(
                        sol_y[4*j][0], sol_y[4*j+1][0],
                        label="r{0}".format(j-(ROBOT_NUM-1)), marker="^", color=color_list[j-(ROBOT_NUM-1)]
                    )
                    v_center_s.append(p)


            pair_s = []
            for j in range(ROBOT_NUM):
                for p in pres_pair[j]:
                    k, _ = p
                    frame_x = [sol_y[4*k][0], sol_y[4*j][0]]
                    frame_y = [sol_y[4*k+1][0], sol_y[4*j+1][0]]
                    p, = ax.plot(frame_x, frame_y, color="k")
                    pair_s.append(p)


            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
            ax.set_xlim(mid_x-max_range, mid_x+max_range)
            ax.set_ylim(mid_y-max_range, mid_y+max_range)
            ax.grid()
            ax.set_aspect('equal')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            time_template = 'time = %.2f [s]'

            def update_2d(i):
                #print("i = ", i)
                for j in range(ROBOT_NUM):
                    robot_s[j].set_center([sol_y[4*j][i], sol_y[4*j+1][i]])
                    traj_s[j].set_data(sol_y[4*j][:i], sol_y[4*j+1][:i])
                    center_s[j].set_data(sol_y[4*j][i], sol_y[4*j+1][i])

                if len(sol_y) != 2*TASK_DIM*ROBOT_NUM:  # 仮想ロボットあり
                    for j in range(ROBOT_NUM, 2*ROBOT_NUM-1):
                        #print("  j = ", j)
                        v_robot_s[j-(ROBOT_NUM)].set_center([sol_y[4*j][i], sol_y[4*j+1][i]])
                        v_traj_s[j-(ROBOT_NUM)].set_data(sol_y[4*j][:i], sol_y[4*j+1][:i])
                        v_center_s[j-(ROBOT_NUM)].set_data(sol_y[4*j][i], sol_y[4*j+1][i])


                l = 0
                for j in range(ROBOT_NUM):
                    for p in pres_pair[j]:
                        k, _ = p
                        frame_x = [sol_y[4*k][i], sol_y[4*j][i]]
                        frame_y = [sol_y[4*k+1][i], sol_y[4*j+1][i]]
                        pair_s[l].set_data(frame_x, frame_y)
                        l += 1

                # o_ = o0_ + np.array([[v_* i * time_interval, 0.]]).T
                # o_s.set_offsets([(o_[0,0], o_[1,0])])
                # o_c_s[0].set_center([o_[0,0], o_[1,0]])

                ax.set_title(time_template % sol_t[i])
                return
            
            #t0 = time.perf_counter()
            ani = anm.FuncAnimation(
                fig = fig,
                func = update_2d,
                frames = range(0, len(sol_t), step),
                interval=60
            )
            
            ani.save("{0}/fig/animation/GIF/{1}/{2}.gif".format(dir_base, sim_name, index), writer="pillow")
            ani.save("{0}/fig/animation/MP4/{1}/{2}.mp4".format(dir_base, sim_name, index), writer="ffmpeg")
            plt.clf(); plt.close()
        
    #     else:  ## 3次元 ###################################################################
    #         #obs ball に使う数列
    #         u = np.linspace(0, 2*pi, 30)
    #         v = np.linspace(0, pi, 30)
    #         ball_x_o = COLLISION_R * np.outer(np.cos(u), np.sin(v))
    #         ball_y_o = COLLISION_R * np.outer(np.sin(u), np.sin(v))
    #         ball_z_o = COLLISION_R * np.outer(np.ones(np.size(u)), np.cos(v))
            
    #         fig = plt.figure(figsize=(10,10))
    #         ax = fig.add_subplot(projection="3d")
    #         for i in range(ROBOT_NUM):
    #             ax.plot(
    #                 sol_y[2*TASK_DIM*i], sol_y[2*TASK_DIM*i+1], sol_y[2*TASK_DIM*i+2], 
    #                 label="r{0}".format(i), color=color_list[i]
    #             )
    #             ax.scatter(
    #                 [sol_y[2*TASK_DIM*i][-1]], [sol_y[2*TASK_DIM*i+1][-1]], [sol_y[2*TASK_DIM*i+2][-1]], 
    #                 label="r{0}".format(i), color=color_list[i]
    #             )

    #         for j in range(ROBOT_NUM):
    #             for k in pres_pair[j]:
    #                 frame_x = [sol_y[2*TASK_DIM*k+0][-1], sol_y[2*TASK_DIM*j+0][-1]]
    #                 frame_y = [sol_y[2*TASK_DIM*k+1][-1], sol_y[2*TASK_DIM*j+1][-1]]
    #                 frame_z = [sol_y[2*TASK_DIM*k+2][-1], sol_y[2*TASK_DIM*j+2][-1]]
    #                 ax.plot(frame_x, frame_y, frame_z, color="k")

    #         for i, g in enumerate(xg_s):
    #             if g is not None:
    #                 ax.scatter(
    #                     [g[0,0]], [g[1,0]], [g[2,0]], 
    #                     marker="*", color=color_list[i], label="g{0}".format(i)
    #                 )
            
    #         if len(xo_s) != 0:
    #             for xo in xo_s:
    #                 _x = ball_x_o + xo[0,0]
    #                 _y = ball_y_o + xo[1,0]
    #                 _z = ball_z_o + xo[2,0]
    #                 ax.plot_surface(
    #                     _x, _y, _z,
    #                     color="C7", alpha=0.3,rcount=100, ccount=100, antialiased=False,
    #                 )
            
    #         ax.scatter(
    #             _xo_con[:, 0], _xo_con[:, 1], _xo_con[:, 2],
    #             marker="+", color="k", label="obs"
    #         )

    #         ax.set_title("t = {0}, and {1}".format(sol_t[-1], sol.success))
    #         ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    #         ax.set_xlim(mid_x-max_range, mid_x+max_range)
    #         ax.set_ylim(mid_y-max_range, mid_y+max_range)
    #         ax.set_zlim(mid_z-max_range, mid_z+max_range)
    #         ax.grid();ax.set_aspect('equal'); ax.legend()
    #         fig.savefig("{0}/fig/trajectry/{1}-{2}.jpg".format(dir_base, sim_name, index))
    #         plt.clf(); plt.close()


    #         ## アニメ ############################################################################

            
    #         fig = plt.figure(figsize=(10,10))
    #         ax = fig.add_subplot(projection="3d")

    #         for i, g in enumerate(xg_s):
    #             if g is not None:
    #                 ax.scatter(
    #                     [g[0,0]], [g[1,0]], [g[2,0]], 
    #                     marker="*", color=color_list[i], label="g{0}".format(i)
    #                 )
            
    #         if len(xo_s) != 0:
    #             for xo in xo_s:
    #                 _x = ball_x_o + xo[0,0]
    #                 _y = ball_y_o + xo[1,0]
    #                 _z = ball_z_o + xo[2,0]
    #                 ax.plot_surface(
    #                     _x, _y, _z, color="C7", 
    #                     alpha=0.3,rcount=100, ccount=100, antialiased=False,
    #                 )
    #         o_s = ax.scatter(
    #             _xo_con[:, 0], _xo_con[:, 1], _xo_con[:, 2], 
    #             marker="+", color="k", label="obs"
    #         )

    #         ball_x_r = ROBOT_R * np.outer(np.cos(u), np.sin(v))
    #         ball_y_r = ROBOT_R * np.outer(np.sin(u), np.sin(v))
    #         ball_z_r = ROBOT_R * np.outer(np.ones(np.size(u)), np.cos(v))
    #         robot_s = []
    #         traj_s = []
    #         for j in range(ROBOT_NUM):
    #             # _x = ball_x_r + sol_y[2*TASK_DIM*j+0][0]
    #             # _y = ball_y_r + sol_y[2*TASK_DIM*j+1][0]
    #             # _z = ball_z_r + sol_y[2*TASK_DIM*j+2][0]
    #             # robot_s.append(ax.plot_surface(_x, _y, _z, color="C7", alpha=0.3,rcount=100, ccount=100, antialiased=False,))
                
    #             scat = ax.scatter(
    #                 [sol_y[2*TASK_DIM*j][0]], [sol_y[2*TASK_DIM*j+1][0]], [sol_y[2*TASK_DIM*j+2][0]], 
    #                 label="r{0}".format(j), color=color_list[j]
    #             )
    #             robot_s.append(scat)
                
    #             p, = ax.plot(
    #                 sol_y[2*TASK_DIM*j][:0], sol_y[2*TASK_DIM*j+1][:0], sol_y[2*TASK_DIM*j+2][:0], 
    #                 label="r{0}".format(j), color=color_list[j]
    #             )
    #             traj_s.append(p)

    #         pair_s = []
    #         for j in range(ROBOT_NUM):
    #             for k in pres_pair[j]:
    #                 frame_x = [sol_y[2*TASK_DIM*k+0][0], sol_y[2*TASK_DIM*j+0][0]]
    #                 frame_y = [sol_y[2*TASK_DIM*k+1][0], sol_y[2*TASK_DIM*j+1][0]]
    #                 frame_z = [sol_y[2*TASK_DIM*k+2][0], sol_y[2*TASK_DIM*j+2][0]]
    #                 p, = ax.plot(frame_x, frame_y, frame_z, color="k")
    #                 pair_s.append(p)


    #         ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    #         ax.set_xlim(mid_x-max_range, mid_x+max_range)
    #         ax.set_ylim(mid_y-max_range, mid_y+max_range)
    #         ax.set_zlim(mid_z-max_range, mid_z+max_range)
    #         ax.grid(); ax.set_aspect('equal'); ax.legend()
    #         time_template = 'time = %.2f [s]'

    #         scale = 10
    #         f_scale = 0.1

    #         def update_3d(i):
    #             for j in range(ROBOT_NUM):
    #                 #robot_s[j].remove()
    #                 _x = ball_x_r + sol_y[2*TASK_DIM*j+0][0]
    #                 _y = ball_y_r + sol_y[2*TASK_DIM*j+1][0]
    #                 _z = ball_z_r + sol_y[2*TASK_DIM*j+2][0]
    #                 #robot_s[j] = ax.plot_surface(_x, _y, _z, color="C7", alpha=0.3,rcount=100, ccount=100, antialiased=False,)

    #                 traj_s[j].set_data(sol_y[2*TASK_DIM*j][:i], sol_y[2*TASK_DIM*j+1][:i])
    #                 traj_s[j].set_3d_properties(sol_y[2*TASK_DIM*j+2][:i])
                    
    #                 #print((sol_y[2*TASK_DIM*j][i], sol_y[2*TASK_DIM*j+1][i], sol_y[2*TASK_DIM*j+2][i]))
    #                 #robot_s[j]._offsets3d = (sol_y[2*TASK_DIM*j][i], sol_y[2*TASK_DIM*j+1][i], sol_y[2*TASK_DIM*j+2][i])

    #             l = 0
    #             for j in range(ROBOT_NUM):
    #                 for k in pres_pair[j]:
    #                     frame_x = [sol_y[2*TASK_DIM*k+0][i], sol_y[2*TASK_DIM*j+0][i]]
    #                     frame_y = [sol_y[2*TASK_DIM*k+1][i], sol_y[2*TASK_DIM*j+1][i]]
    #                     frame_z = [sol_y[2*TASK_DIM*k+2][i], sol_y[2*TASK_DIM*j+2][i]]
    #                     pair_s[l].set_data(frame_x, frame_y)
    #                     pair_s[l].set_3d_properties(frame_z)
    #                     l += 1



    #             ax.set_title(time_template % sol_t[i])
    #             return
            
    #         ani = anm.FuncAnimation(
    #             fig = fig,
    #             func = update_3d,
    #             frames = range(0, len(sol_t), step),
    #             interval=60
    #         )
            
    #         ani.save("{0}/fig/animation/GIF/{1}-{2}.gif".format(dir_base, sim_name, index), writer="pillow")
    #         ani.save("{0}/fig/animation/MP4/{1}-{2}.mp4".format(dir_base, sim_name, index), writer="ffmpeg")
    #         plt.show()
    #         plt.clf(); plt.close()
    
    # print("simulation {0} done!".format(index))


def runner(sim_path, sim_param):
    date_now = datetime.datetime.now()
    today_label = date_now.strftime('%Y-%m-%d')
    os.makedirs("../syuron/point/" + today_label, exist_ok=True)
    data_label = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    dir_base = "../syuron/point/{0}/{1}".format(today_label, data_label)
    os.makedirs(dir_base, exist_ok=True)

    for s in ["rmp", "fabric"]:
        os.makedirs(dir_base + "/csv/" + s, exist_ok=True)
        os.makedirs(dir_base + "/fig/trajectry/" + s, exist_ok=True)
        os.makedirs(dir_base + "/fig/animation/GIF/" + s, exist_ok=True)
        os.makedirs(dir_base + "/fig/animation/MP4/" + s, exist_ok=True)
        os.makedirs(dir_base + "/fig/state/" + s, exist_ok=True)
        os.makedirs(dir_base + "/message/" + s, exist_ok=True)
        os.makedirs(dir_base + "/score/" + s, exist_ok=True)
        os.makedirs(dir_base + "/time/" + s, exist_ok=True)
    
    os.makedirs(dir_base + "/condition", exist_ok=True)
    os.makedirs(dir_base + "/config", exist_ok=True)

    if sim_path is not None:
        assert sim_param is not None
        with open(sim_path, "r") as f:
            sim_param = yaml.safe_load(f)
    
    with open(dir_base + "/config/config.yaml", 'w') as f:
        yaml.dump(sim_param, f)
    
    trial = sim_param["trial"]  # 並列実行数
    
    if trial == 1:
        test(dir_base, sim_param, trial, np.random.RandomState(np.random.randint(0, 10000000)))
    else:
        itr = [
            (dir_base, sim_param, i, np.random.RandomState(np.random.randint(0, 10000000)))
            for i in range(trial)
        ]
        core = cpu_count()
        with Pool(core) as p:
            result = p.starmap(func=test, iterable=itr)


if __name__ == "__main__":
    sim_path = "/home/matsuta_conda/src/rmp-py/config_syuron/point_2d.yaml"
    #runner(sim_path)
    
    runner(sim_path=None, sim_param=p2d.sim_param)
    #runner(sim_path=None, sim_param=p3d.sim_param)

