import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from scipy import integrate
from functools import lru_cache

import rmp_tree
import rmp_leaf
import mappings
import robot_model_sice
import visualization

TIME_SPAN = 60*3
TIME_INTERVAL = 1e-2
q0 = np.array([[np.pi/2, 0, 0, 0]]).T
#q0 = np.array([[-np.pi*2/4, np.pi*2.8/4, 0, 0]]).T
#q0_dot = np.array([[0.1, 0, 0, 0]]).T
q0_dot = np.zeros_like(q0)

r = rmp_tree.Root(
    x0 = q0,
    x0_dot = q0_dot
)



n1 = rmp_tree.LeafBase(
    name="x1", parent=r, dim=2,
    mappings=robot_model_sice.X1()
)
r.add_child(n1)

n2 = rmp_tree.LeafBase(
    name="x2", parent=r, dim=2,
    mappings=robot_model_sice.X2()
)
r.add_child(n2)

n3 = rmp_tree.LeafBase(
    name="x3", parent=r, dim=2,
    mappings=robot_model_sice.X3()
)
r.add_child(n3)

x4 = robot_model_sice.X4()
n4 = rmp_tree.Node(
    name="x4", parent=r, dim=2,
    mappings=x4
)
r.add_child(n4)



g = np.array([[2.0, 1.0]]).T
#g = np.array([[0.1, 0]]).T
g_dot = np.zeros_like(g)


attracter = rmp_leaf.GoalAttractor(
    name="ee-attractor", parent=n4, dim=2,
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
n4.add_child(attracter)


jl = rmp_leaf.JointLimitAvoidance(
    name="jl", parent=r,
    calc_mappings=mappings.Id(),
    gamma_p = 0.01,
    gamma_d = 0.05,
    lam = 1,
    sigma = 0.1,
    q_max = robot_model_sice.q_max,
    q_min = robot_model_sice.q_min,
    q_neutral = robot_model_sice.q_neutral
)
r.add_child(jl)


# ### 木構造について確認 ###

# r.print_all_state()
# print("-"*100)
# print("pushforward!")
# r.pushforward()
# r.print_all_state()
# print("-"*100)
# print("pullback!")
# r.pullback()
# r.print_all_state()
# print("-"*100)
# print("resolve!")
# r.resolve()
# print("-"*100)



# ### 実行 ###
# times = np.arange(0, TIME_SPAN, TIME_INTERVAL)
# for i, t in enumerate(times):
#     print("\ni = ", i)
#     if i == 0:
#         print("t = ", t)
#         q_list = [q0]
#         q_dot_list = [q0_dot]
#         r.pushforward()
#         error = [n4.x]
#     else:
#         print("t = ", t)
#         r.pushforward()
#         r.pullback()
#         r.resolve()
        
        
#         q_list.append(r.x.copy())
#         q_dot_list.append(r.x_dot.copy())
#         error.append(n4.x.copy())
#         #r.print_all_state()
#         #r.print_state()

#     #print(q_list)


# q_list = np.concatenate(q_list, axis=1)
# q_dot_list = np.concatenate(q_dot_list, axis=1)
# error= np.concatenate(error, axis=1)


# #print(q_list)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(times, q_list[0, :], label="q1", marker="")
# ax.plot(times, q_list[1, :], label="q2", marker="")
# ax.plot(times, q_list[2, :], label="q3", marker="")
# ax.plot(times, q_list[3, :], label="q4", marker="")
# ax.legend()
# ax.grid()
# fig.savefig("test.png")



# fig2 = plt.figure()
# ax2 = fig2.add_subplot()
# ax2.plot(error[0, :], error[1, :], label="ee")
# ax2.scatter(g[0,0], g[1,0], label="goal", marker="*", color="red")
# ax2.legend()
# ax2.grid()
# ax2.set_aspect('equal')
# fig2.savefig("ee.png")




### scipy使用 ###

#@lru_cache()
def dX(t, X):
    print("\nt = ", t)
    X = X.reshape(-1, 1)
    q_ddot = r.solve(q=X[:4, :], q_dot=X[4:, :])
    X_dot = np.concatenate([X[4:, :], q_ddot])
    return np.ravel(X_dot)




def dX2(t, X):
    """いつもの"""
    print("\nt = ", t)
    X = X.reshape(-1, 1)
    q = X[:4, :]
    q_dot = X[4:, :]
    
    root_f = np.zeros((4, 1))
    root_M = np.zeros((4, 4))
    
    jl.set_state(q, q_dot)
    jl.calc_rmp_func()
    root_f += jl.f
    root_M += jl.M
    
    ee_x = x4.phi(q)
    ee_J = x4.J(q)
    ee_x_dot = x4.velocity(ee_J, q_dot)
    ee_J_dot = x4.J_dot(q, q_dot)
    attracter.set_state(ee_x-g, ee_x_dot-g_dot)
    attracter.calc_rmp_func()
    root_f += ee_J.T @ (attracter.f - attracter.M @ ee_J_dot @ q_dot)
    root_M += ee_J.T @ attracter.M @ ee_J
    
    q_ddot = LA.pinv(root_M) @ root_f
    
    X_dot = np.concatenate([q_dot, q_ddot])
    return np.ravel(X_dot)






sol = integrate.solve_ivp(
    #fun = dX,
    fun = dX2,
    t_span = (0, TIME_SPAN),
    y0 = np.ravel(np.concatenate([q0, q0_dot])),
    t_eval=np.arange(0, TIME_SPAN, TIME_INTERVAL),
    #atol=1e-6
)

print(sol.message)

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 13))
# axes[0].plot(sol.t, sol.y[0], label="q1")
# axes[0].plot(sol.t, sol.y[1], label="q2")
# axes[0].plot(sol.t, sol.y[2], label="q3")
# axes[0].plot(sol.t, sol.y[3], label="q4")

# _x = []
# _y = []
# for i in range(len(sol.t)):
#     _X = x4.phi(np.array([[sol.y[0][i],sol.y[1][i],sol.y[2][i],sol.y[3][i]]]).T)
#     _x.append(_X[0,0])
#     _y.append(_X[1,0])
# axes[1].plot(_x, _y, label="ee")
# axes[1].scatter(g[0,0], g[1,0], marker="*", color="red")
# axes[1].set_aspect('equal')

# for ax in axes.ravel():
#     ax.legend()
#     ax.grid()
# fig.savefig("solver0.png")



def x0(q):
    return np.zeros((2, 1))
x1_map = robot_model_sice.X1()
x2_map = robot_model_sice.X2()
x3_map = robot_model_sice.X3()
x4_map = robot_model_sice.X4()

q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
    q_s = [sol.y[0], sol.y[1], sol.y[2], sol.y[3]],
    joint_phi_s=[x0, x1_map.phi, x2_map.phi, x3_map.phi, x4_map.phi],
    is3D=False,
    ee_phi=x4_map.phi
)




ani = visualization.make_animation(
    t_data = sol.t,
    joint_data=joint_data,
    is3D=False,
    goal_data=np.array([[g[0,0], g[1,0]]*len(sol.t)]).reshape(len(sol.t), 2)
)


plt.show()