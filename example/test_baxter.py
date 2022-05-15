import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from scipy import integrate

import sys
sys.path.append('.')
import environment

# from functools import lru_cache
# from numba import njit

import rmp_tree
import rmp_leaf
import mappings
import visualization


import baxter.baxter as baxter

TIME_SPAN = 60
TIME_INTERVAL = 1e-2

q0 = baxter.Common.q_neutral  #初期値
q0_dot = np.zeros_like(q0)

r = rmp_tree.Root(dim=7, isMulti=True)
r.set_state(q0, q0_dot)


### 関節角度制限 ###
jl = rmp_leaf.JointLimitAvoidance(
    name="jl",
    parent=r,
    calc_mappings=mappings.Identity(),
    gamma_p = 0.01,
    gamma_d = 0.05,
    lam = 1,
    sigma = 0.1,
    q_max = baxter.Common.q_max,
    q_min = baxter.Common.q_min,
    q_neutral = baxter.Common.q_neutral
)
r.add_child(jl)


# tree construction
ns: list[list[rmp_tree.Node]] = []
for i, rs in enumerate(baxter.Common.R_BARS_ALL[:-1]):
    n_: list[rmp_tree.Node] = []
    for j, _ in enumerate(rs):
        n_.append(
            rmp_tree.Node(
                name = 'x_' + str(i) + '_' + str(j),
                dim = 3,
                parent = r,
                mappings = baxter.CPoint(i, j)
            )
        )
    ns.append(n_)

for n_ in ns:
    for n in n_:
        r.add_child(n)



# end-effector node
n_ee = rmp_tree.Node(
    name = "ee",
    dim = 3,
    parent = r,
    mappings = baxter.CPoint(7, 0)
)
r.add_child(n_ee)


### 目標 ###
g = np.array([[0.4, -0.4, 1]]).T
g_dot = np.zeros_like(g)

attracter = rmp_leaf.GoalAttractor(
    name="ee-attractor", parent=n_ee, dim=3,
    calc_mappings=mappings.Translation(g, g_dot),
    max_speed = 5.0,
    gain = 5.0,
    f_alpha = 0.15,
    sigma_alpha = 1.0,
    sigma_gamma = 1.0,
    wu = 10.0,
    wl = 0.1,
    alpha = 0.15,
    epsilon = 0.5,
)
n_ee.add_child(attracter)


### 障害物 ###
o_s = environment._set_cylinder(
    r=0.1, L=1, x=0.2, y=-0.4, z=1, n=100, alpha=0, beta=0, gamma=90
)
for n in ns:
    for m_ in n:
        for i, o in enumerate(o_s):
            obs_node = rmp_leaf.ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + m_.name,
                parent=m_,
                calc_mappings=mappings.Distance(o, np.zeros_like(o)),
                scale_rep=0.2,
                scale_damp=1,
                gain=5,
                sigma=1,
                rw=0.15
            )
            m_.add_child(obs_node)

for o in o_s:
    obs_node = rmp_leaf.ObstacleAvoidance(
        name="obs",
        parent=n_ee,
        calc_mappings=mappings.Distance(o, np.zeros_like(o)),
        scale_rep=0.2,
        scale_damp=1,
        gain=5,
        sigma=1,
        rw=0.15
    )
    n_ee.add_child(obs_node)


def dX(t, X):
    print("\nt = ", t)
    X = X.reshape(-1, 1)
    q_ddot = r.solve(q=X[:7, :], q_dot=X[7:, :])
    X_dot = np.concatenate([X[7:, :], q_ddot])
    return np.ravel(X_dot)


sol = integrate.solve_ivp(
    fun = dX,
    #fun = dX2,
    t_span = (0, TIME_SPAN),
    y0 = np.ravel(np.concatenate([q0, q0_dot])),
    t_eval=np.arange(0, TIME_SPAN, TIME_INTERVAL),
    #atol=1e-6
)
print(sol.message)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 13))
for i in range(7):
    axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
    axes[1].plot(sol.t, sol.y[i+7], label="dq" + str(i))

for i in range(2):
    axes[i].legend()
    axes[i].grid()


fig.savefig("solver_bax_2.png")



def x0(q):
    return np.zeros((3, 1))

q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
    q_s = [sol.y[i] for i in range(7)],
    joint_phi_s=[x0, baxter.o_W0, baxter.o_BR, baxter.o_0, baxter.o_1, baxter.o_2, baxter.o_3, baxter.o_4, baxter.o_5, baxter.o_6, baxter.o_ee],
    is3D=True,
    ee_phi=baxter.o_ee
)




ani = visualization.make_animation(
    t_data = sol.t,
    joint_data=joint_data,
    is3D=True,
    goal_data=np.array([[g[0,0], g[1,0], g[2,0]]*len(sol.t)]).reshape(len(sol.t), 3),
    obs_data=o_s,
    save_dir_path="pic/",
    isSave=True
)




plt.show()