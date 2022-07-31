""""""

import numpy as np
from numpy.typing import NDArray
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


import baxter.baxter as baxter

TIME_SPAN = 20
TIME_INTERVAL = 1e-2

q0 = baxter.Common.q_neutral  #初期値
q0_dot = np.zeros_like(q0)

### 目標 ###
#g = np.array([[0.0, -0.4, 1]]).T
g = np.array([[-0.2, -0.6, 0.99]]).T

g_dot = np.zeros_like(g)



# def main(isMulti: bool, obs_num: int):
#     date_now = datetime.datetime.now()
#     name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
#     base = "../rmp-py_result/" + name + "/"
#     os.makedirs(base, exist_ok=True)
    


#     print("constructing rmp-tree...")
#     t0 = time.perf_counter()
#     r = rmp_node.Root(7, isMulti)
#     r.set_state(q0, q0_dot)


#     ### 関節角度制限 ###
#     jl = rmp_leaf.JointLimitAvoidance(
#         name="jl",
#         parent=r,
#         calc_mappings=mappings.Identity(),
#         gamma_p = 0.1,
#         gamma_d = 0.05,
#         lam = 1,
#         sigma = 0.1,
#         q_max = baxter.Common.q_max,
#         q_min = baxter.Common.q_min,
#         q_neutral = baxter.Common.q_neutral
#     )
#     r.add_child(jl)


#     # tree construction
#     ns: list[list[rmp_node.Node]] = []
#     cpoint_phis = []
#     for i, rs in enumerate(baxter.Common.R_BARS_ALL[:-1]):
#         n_= []
#         for j, _ in enumerate(rs):
#             map_ = baxter.CPoint(i, j)
#             n_.append(
#                 rmp_node.Node(
#                     name = 'x_' + str(i) + '_' + str(j),
#                     dim = 3,
#                     parent = r,
#                     mappings = map_
#                 )
#             )
#             cpoint_phis.append(map_.phi)
#         ns.append(n_)

#     for n_ in ns:
#         for n in n_:
#             r.add_child(n)


#     print("t0.5 = ", time.perf_counter() - t0)
#     # end-effector node
#     map_ = baxter.CPoint(7, 0)
#     n_ee = rmp_node.Node(
#         name = "ee",
#         dim = 3,
#         parent = r,
#         mappings = map_
#     )
#     cpoint_phis.append(map_.phi)
#     r.add_child(n_ee)




#     attracter = rmp_leaf.GoalAttractor(
#         name="ee-attractor", parent=n_ee, dim=3,
#         calc_mappings=mappings.Translation(g, g_dot),
#         max_speed = 5.0,
#         gain = 5.0,
#         f_alpha = 0.15,
#         sigma_alpha = 1.0,
#         sigma_gamma = 1.0,
#         wu = 10.0,
#         wl = 0.1,
#         alpha = 0.15,
#         epsilon = 0.5,
#     )
#     n_ee.add_child(attracter)


#     ### 障害物 ###
#     o_s = environment._set_cylinder(
#         r=0.1, L=0.8, x=-0.2, y=-0.4, z=0.8, n=obs_num, alpha=0, beta=0, gamma=90
#     )
#     for n in ns:
#         for m_ in n:
#             for i, o in enumerate(o_s):
#                 obs_node = rmp_leaf.ObstacleAvoidance(
#                     name="obs_" + str(i) + ", at " + m_.name,
#                     parent=m_,
#                     calc_mappings=mappings.Distance(o, np.zeros_like(o)),
#                     scale_rep=0.2,
#                     scale_damp=1,
#                     gain=5,
#                     sigma=1,
#                     rw=0.15
#                 )
#                 m_.add_child(obs_node)

#     print("t1 = ", time.perf_counter() - t0)
#     for o in o_s:
#         obs_node = rmp_leaf.ObstacleAvoidance(
#             name="obs",
#             parent=n_ee,
#             calc_mappings=mappings.Distance(o, np.zeros_like(o)),
#             scale_rep=0.2,
#             scale_damp=1,
#             gain=5,
#             sigma=1,
#             rw=0.2
#         )
#         n_ee.add_child(obs_node)


#     print("construct rmp-tree done! time = ", time.perf_counter() - t0)


#     def dX(t, X):
#         print("\nt = ", t)
#         X = X.reshape(-1, 1)
#         q_ddot = r.solve(q=X[:7, :], q_dot=X[7:, :])
#         X_dot = np.concatenate([X[7:, :], q_ddot])
#         #print(jl.name, "  ", jl.f.T)
#         return np.ravel(X_dot)
    
    
    
#     sol = integrate.solve_ivp(
#         fun = dX,
#         #fun = dX2,
#         t_span = (0, TIME_SPAN),
#         y0 = np.ravel(np.concatenate([q0, q0_dot])),
#         t_eval=np.arange(0, TIME_SPAN, TIME_INTERVAL),
#         #atol=1e-6
#     )
#     print(sol.message)
    
    
#     ### 以下グラフ化 ###
#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 13))
#     for i in range(7):
#         axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
#         axes[1].plot(sol.t, sol.y[i+7], label="dq" + str(i))
#     for i in range(2):
#         axes[i].legend()
#         axes[i].grid()
#     fig.savefig(base+"solver_bax_2.png")

#     def x0(q):
#         return np.zeros((3, 1))


#     q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
#         q_s = [sol.y[i] for i in range(7)],
#         cpoint_phi_s=cpoint_phis,
#         joint_phi_s=[x0, baxter.o_W0, baxter.o_BR, baxter.o_0, baxter.o_1, baxter.o_2, baxter.o_3, baxter.o_4, baxter.o_5, baxter.o_6, baxter.o_ee],
#         is3D=True,
#         ee_phi=baxter.o_ee
#     )

#     ani = visualization.make_animation(
#         t_data = sol.t,
#         joint_data=joint_data,
#         cpoint_data=cpoint_data,
#         is3D=True,
#         goal_data=np.array([[g[0,0], g[1,0], g[2,0]]*len(sol.t)]).reshape(len(sol.t), 3),
#         obs_data=o_s,
#         save_path=base+"animation.gif",
#         isSave=True
#     )

#     return sol, ani



def main2(isMulti: bool, obs_num: int):
    """ノードをプロセス毎に再構築"""
    
    date_now = datetime.datetime.now()
    name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
    base = "../rmp-py_result/" + name + "/"
    os.makedirs(base, exist_ok=True)
    
    r = rmp_node.Root(7, isMulti)
    r.set_state(q0, q0_dot)


    ### 障害物 ###
    o_s = environment._set_cylinder(
        r=0.1, L=0.8, x=-0.2, y=-0.4, z=0.8, n=obs_num, alpha=0, beta=0, gamma=90
    )

    def dX(t, X: NDArray[np.float64]):
        print("\nt = ", t)
        X = X.reshape(-1, 1)
        q_ddot = tree_constructor.solve(q=X[:7, :], q_dot=X[7:, :], g=g, o_s=o_s)
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
    for i, rs in enumerate(baxter.Common.R_BARS_ALL[:-1]):
        for j, _ in enumerate(rs):
            map_ = baxter.CPoint(i, j)
            cpoint_phis.append(map_.phi)

    map_ = baxter.CPoint(7, 0)
    cpoint_phis.append(map_.phi)


    def x0(q):
        return np.zeros((3, 1))

    q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
        q_s = [sol.y[i] for i in range(7)],
        cpoint_phi_s=cpoint_phis,
        joint_phi_s=[x0, baxter.o_W0, baxter.o_BR, baxter.o_0, baxter.o_1, baxter.o_2, baxter.o_3, baxter.o_4, baxter.o_5, baxter.o_6, baxter.o_ee],
        is3D=True,
        ee_phi=baxter.o_ee
    )

    ani = visualization.make_animation(
        t_data = sol.t,
        joint_data=joint_data,
        cpoint_data=cpoint_data,
        is3D=True,
        goal_data=np.array([[g[0,0], g[1,0], g[2,0]]*len(sol.t)]).reshape(len(sol.t), 3),
        obs_data=o_s,
        save_path=base+"animation.gif",
        isSave=True,
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


#main2(10)
#main2(100)
# main2(500)
runner(300)



# plt.show()