import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from scipy import integrate

import rmp_tree
import rmp_leaf
import mappings
import visualization
import sys
sys.path.append('.')
import robot_model_baxter as baxter

TIME_SPAN = 60
TIME_INTERVAL = 1e-2
q0 = baxter.Common.q_neutral
q0_dot = np.zeros_like(q0)

r = rmp_tree.Root(
    x0 = q0,
    x0_dot = q0_dot
)

# tree construction
ns = []
for i, rs in enumerate(baxter.Common.R_BARS_ALL[:-1]):
    n_ = []
    for j, _ in enumerate(rs):
        n_.append(
            rmp_tree.LeafBase(
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

g = np.array([[0.3, -0.6, 1]]).T
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


jl = rmp_leaf.JointLimitAvoidance(
    name="jl",
    parent=r,
    calc_mappings=mappings.Id(),
    gamma_p = 0.01,
    gamma_d = 0.05,
    lam = 1,
    sigma = 0.1,
    q_max = baxter.Common.q_max,
    q_min = baxter.Common.q_min,
    q_neutral = baxter.Common.q_neutral
)
r.add_child(jl)



### scipy使用 ###

#@lru_cache()
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


fig.savefig("solver_bax.png")



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
    epoch_max=500,
    goal_data=np.array([[g[0,0], g[1,0], g[2,0]]*len(sol.t)]).reshape(len(sol.t), 3),
    save_dir_path=""
)




plt.show()