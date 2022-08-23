"""franka emikaロボットのテスト"""


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




ROBOT_NAME = 'franka_emika'
TIME_SPAN = 60
TIME_INTERVAL = 1e-2/2

q0 = franka_emika.CPoint.q_neutral  #初期値
q0_dot = np.zeros_like(q0)

### 目標 ###
g = np.array([[0.5, -0.08, 0.5]]).T

g_dot = np.zeros_like(g)




def main2(isMulti: bool, obs_num: int):
    """ノードをプロセス毎に再構築"""
    
    date_now = datetime.datetime.now()
    name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    base = "../rmp-py_result/" + name + "/"
    os.makedirs(base, exist_ok=True)
    
    r = rmp_node.Root(7, isMulti)
    r.set_state(q0, q0_dot)


    rmp_param: dict = {
        'joint_limit_avoidance' : {
            'gamma_p' : 0.05,
            'gamma_d' : 0.05,
            'lam' : 1,
            'sigma' : 0.1
        },
        'attractor' : {
            'max_speed' : 8.0,
            'gain' : 5.0,
            'f_alpha' : 0.15,
            'sigma_alpha' : 1.0,
            'sigma_gamma' : 1.0,
            'wu' : 10.0,
            'wl' : 0.1,
            'alpha' : 0.15,
            'epsilon' : 0.5,
        },
        'obstacle_avoidance' : {
            'scale_rep' : 0.1,
            'scale_damp' : 1,
            'gain' : 5,
            'sigma' : 1,
            'rw' : 0.1
        }
    }


    ### 障害物 ###
    o_s = environment.set_cylinder(
        r=0.05, L=1.2, x=0.3, y=-0.16+0.05, z=0.6, n=obs_num, alpha=0, beta=0, gamma=90
    )
    print(type(o_s))
    o_s += environment.set_cylinder(
        r=0.05, L=1.2, x=0.3, y=0.16+0.05, z=0.6, n=obs_num, alpha=0, beta=0, gamma=90
    )
    #print(o_s)

    def dX(t, X):
        print("\nt = ", t)
        X = X.reshape(-1, 1)
        q_ddot = tree_constructor.solve(
            q=X[:7, :], q_dot=X[7:, :], g=g, o_s=o_s,
            robot_name=ROBOT_NAME,
            rmp_param=rmp_param
        )
        print("ddq = ", q_ddot.T)
        X_dot = np.concatenate([X[7:, :], q_ddot])
        return np.ravel(X_dot)
    
    
    sol = integrate.solve_ivp(
        fun = dX,
        t_span = (0, TIME_SPAN),
        y0 = np.ravel(np.concatenate([q0, q0_dot])),
        t_eval=np.arange(0, TIME_SPAN, TIME_INTERVAL),
        #atol=1e-10
    )
    print(sol.message)
    
    
    ### 以下グラフ化 ###
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 13))
    for i in range(7):
        axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
        axes[1].plot(sol.t, sol.y[i+7], label="dq" + str(i))
    for i in range(2):
        axes[i].legend()
        axes[i].grid()
    fig.savefig(base+"solver_bax_2.png")


    cpoint_phis = []
    for i, rs in enumerate(franka_emika.CPoint.R_BARS_ALL):#[:-1]):
        for j, _ in enumerate(rs):
            map_ = franka_emika.CPoint(i, j)
            cpoint_phis.append(map_.phi)

    map_ = franka_emika.CPoint(7, 0)
    cpoint_phis.append(map_.phi)


    def x0(q):
        return np.zeros((3, 1))

    q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
        q_s = [sol.y[i] for i in range(7)],
        cpoint_phi_s=cpoint_phis,
        joint_phi_s=[x0, franka_emika.o_0, franka_emika.o_1, franka_emika.o_2, franka_emika.o_3, franka_emika.o_4, franka_emika.o_5, franka_emika.o_6, franka_emika.o_ee],
        is3D=True,
        ee_phi=franka_emika.o_ee
    )

    ani = visualization.make_animation(
        t_data = sol.t,
        joint_data=joint_data,
        cpoint_data=cpoint_data,
        is3D=True,
        goal_data=np.array([[g[0,0], g[1,0], g[2,0]]*len(sol.t)]).reshape(len(sol.t), 3),
        obs_data=np.concatenate(o_s, axis=1).T,
        save_path=base+"animation.gif",
        isSave=True,
        #epoch_max=120
    )
    
    return sol, ani


def runner(obs):
    print("障害物の個数 :", obs)

    # print("並列化無し")
    # t0 = time.process_time()
    # t1 = time.perf_counter()
    # _, ani = main(False, obs)
    # print("cpu time = ", time.process_time() - t0)
    # print("real time = ", time.perf_counter() - t1)


    print("並列化有り")
    t0 = time.process_time()
    t1 = time.perf_counter()
    _, ani2 = main2(True, obs)
    print("cpu time = ", time.process_time() - t0)
    print("real time = ", time.perf_counter() - t1)
    
    plt.show()


if __name__ == "__main__":
    #main2(10)
    #main2(100)
    # main2(500)
    runner(200)



    # plt.show()