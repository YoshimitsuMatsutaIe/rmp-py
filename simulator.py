"""シミュレーション"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate
from typing import Union
import datetime
import os
#from pathlib import Path
import sys
sys.path.append('.')
import environment

# from functools import lru_cache
# from numba import njit

import rmp_node
import rmp_leaf
import tree_constructor
import mappings
import visualization


import franka_emika.franka_emika as franka_emika
import baxter.baxter as baxter





def dx(t, x, g, o_s, robot_name, rmp_param):
    """ODE"""
    print("\nt = ", t)
    x = x.reshape(-1, 1)
    q_ddot = tree_constructor.solve(
        q=x[:7, :], q_dot=x[7:, :], g=g, o_s=o_s,
        robot_name=robot_name,
        rmp_param=rmp_param
    )
    print("ddq = ", q_ddot.T)
    x_dot = np.concatenate([x[7:, :], q_ddot])
    return np.ravel(x_dot)


def main(param: dict):
    
    date_now = datetime.datetime.now()
    name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    base = "../rmp-py_result/" + name + "/"
    os.makedirs(base, exist_ok=True)
    
    
    env = param["env_param"]
    obstacle = []
    for obs_param in env["obstacle"].values():
        if obs_param["name"] == "cylinder":
            obstacle += environment._set_cylinder(**obs_param["param"])
        else:
            assert False
    
    goal = np.array([[
        env["goal"]["param"]["x"],
        env["goal"]["param"]["y"],
        env["goal"]["param"]["z"],
    ]]).T
    
    
    if param["robot_name"] == "baxter":
        robot_model = baxter
    elif param["robot_name"] == "franka_emika":
        robot_model = franka_emika
    else:
        assert False
    
    sol = integrate.solve_ivp(
        fun = dx,
        t_span = (0, param["time_span"]),
        y0 = np.ravel(np.concatenate([robot_model.CPoint.q_neutral, np.zeros_like(robot_model.CPoint.q_neutral)])),
        t_eval=np.arange(0, param["time_span"], param["time_interval"]),
        args=(goal, obstacle, param["rmp_param"])
        #atol=1e-10
    )
    print(sol.message)
    
    

    ### 以下グラフ化 ###
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 13))
    for i in range(robot_model.CPoint.dim):
        axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
        axes[1].plot(sol.t, sol.y[i+7], label="dq" + str(i))
    for i in range(robot_model.CPoint.dim):
        axes[i].legend()
        axes[i].grid()
    fig.savefig(base+"solver_bax_2.png")


    cpoint_phis = []
    for i, rs in enumerate(robot_model.CPoint.R_BARS_ALL):#[:-1]):
        for j, _ in enumerate(rs):
            map_ = robot_model.CPoint(i, j)
            cpoint_phis.append(map_.phi)

    map_ = robot_model.CPoint(7, 0)
    cpoint_phis.append(map_.phi)


    def x0(q):
        return np.zeros((3, 1))

    q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
        q_s = [sol.y[i] for i in range(7)],
        cpoint_phi_s=cpoint_phis,
        joint_phi_s=robot_model.CPoint.JOINT_PHI,
        is3D=True,
        ee_phi=robot_model.o_ee
    )

    ani = visualization.make_animation(
        t_data = sol.t,
        joint_data=joint_data,
        cpoint_data=cpoint_data,
        is3D=True,
        goal_data=np.array([[goal[0,0], goal[1,0], goal[2,0]]*len(sol.t)]).reshape(len(sol.t), 3),
        obs_data=np.concatenate(obstacle, axis=1).T,
        save_path=base+"animation.gif",
        isSave=True,
        #epoch_max=120
    )



if __name__ == "__main__":
    pass
