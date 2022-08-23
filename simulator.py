"""シミュレーション"""

import numpy as np
import matplotlib.pyplot as plt
#import time
from scipy import integrate
#from typing import Union
import datetime
import os
#from pathlib import Path
import shutil
import json
import sys
sys.path.append('.')
import environment

# from functools import lru_cache
# from numba import njit

# import rmp_node
# import rmp_leaf
import tree_constructor
# import mappings
import visualization


import robot_franka_emika.franka_emika as franka_emika
import robot_baxter.baxter as baxter
import robot_sice.sice as sice





def dx(t, x, g, o_s, robot_name, rmp_param):
    """ODE"""
    print("\nt = ", t)
    dim = x.shape[0] // 2
    x = x.reshape(-1, 1)
    q_ddot = tree_constructor.solve(
        q=x[:dim, :], q_dot=x[dim:, :], g=g, o_s=o_s,
        robot_name=robot_name,
        rmp_param=rmp_param
    )
    print("ddq = ", q_ddot.T)
    x_dot = np.concatenate([x[dim:, :], q_ddot])
    return np.ravel(x_dot)



def main(param_path):
    
    date_now = datetime.datetime.now()
    name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    base = "../rmp-py_result/" + name + "/"
    os.makedirs(base, exist_ok=True)
    
    
    with open(param_path) as f:
        param = json.load(f)
    
    shutil.copy2(param_path, base)  # 設定ファイルのコピー作成
    
    
    env = param["env_param"]
    
    ### 障害物 ###
    obstacle = []
    for obs_param in env["obstacle"]:
        if obs_param["type"] == "cylinder":
            obstacle += environment.set_cylinder(**obs_param["param"])
        elif obs_param["type"] == "sphere":
            obstacle += environment.set_sphere(**obs_param["param"])
        elif obs_param["type"] == "box":
            obstacle += environment.set_box(**obs_param["param"])
        elif obs_param["type"] == "cubbie":
            obstacle += environment.set_cubbie(**obs_param["param"])
        elif obs_param["type"] == "point":
            obstacle += environment.set_point(**obs_param["param"])
        else:
            assert False
    
    ### goal ###
    goal = environment.set_point(**env["goal"]["param"])[0]
    
    
    if param["robot_name"] == "baxter":
        robot_model = baxter
    elif param["robot_name"] == "franka_emika":
        robot_model = franka_emika
    elif param["robot_name"] == "sice":
        robot_model = sice
    else:
        assert False
    
    sol = integrate.solve_ivp(
        fun = dx,
        t_span = (0, param["time_span"]),
        y0 = np.ravel(np.concatenate([
            robot_model.CPoint.q_neutral,
            np.zeros_like(robot_model.CPoint.q_neutral)
        ])),
        t_eval=np.arange(0, param["time_span"], param["time_interval"]),
        args=(goal, obstacle, param["robot_name"], param["rmp_param"])
        #atol=1e-10
    )
    print(sol.message)
    
    

    ### 以下グラフ化 ###

    c_dim = robot_model.CPoint.c_dim
    t_dim = robot_model.CPoint.t_dim

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 13))
    for i in range(c_dim):
        axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
        axes[1].plot(sol.t, sol.y[i+c_dim], label="dq" + str(i))
    for i in range(2):
        axes[i].legend()
        axes[i].grid()
        axes[i].set_xlabel("time [s]")
    fig.savefig(base+"solver_bax_2.png")


    cpoint_phis = []
    for i, rs in enumerate(robot_model.CPoint.R_BARS_ALL):
        for j, _ in enumerate(rs):
            map_ = robot_model.CPoint(i, j)
            cpoint_phis.append(map_.phi)

    map_ = robot_model.CPoint(c_dim, 0)
    cpoint_phis.append(map_.phi)



    q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
        q_s = [sol.y[i] for i in range(c_dim)],
        cpoint_phi_s=cpoint_phis,
        joint_phi_s=robot_model.CPoint.JOINT_PHI,
        is3D=True if t_dim==3 else False,
        #ee_phi=robot_model.o_ee
    )

    if t_dim == 3:
        is3D = True
        goal_data = np.array([[goal[0,0], goal[1,0], goal[2,0]]*len(sol.t)]).reshape(len(sol.t), 3)
    elif t_dim == 2:
        is3D = False
        goal_data = np.array([[goal[0,0], goal[1,0],] * len(sol.t)]).reshape(len(sol.t), 2)
    else:
        assert False

    ani = visualization.make_animation(
        t_data = sol.t,
        joint_data=joint_data,
        cpoint_data=cpoint_data,
        is3D=is3D,
        goal_data=goal_data,
        obs_data=np.concatenate(obstacle, axis=1).T,
        save_path=base+"animation.gif",
        isSave=True,
        #epoch_max=120
    )
    
    plt.show()



if __name__ == "__main__":
    pass
