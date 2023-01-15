
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from scipy import integrate
from math import exp
from typing import Union, Tuple
from numba import njit

import mappings
import rmp_node
from rmp_leaf import LeafBase
import fabric


xg = np.array([[2, 3]]).T
m_u = 2
m_l = 0.2
alpha_m = 0.75
k = 10
alpha_psi = 10
k_d = 0

X0 = np.array([0, 0, 0, 0.0])
time_interval = 0.01
time_span = 10
tspan = (0, time_span)
teval = np.arange(0, time_span, time_interval)

attractor = fabric.GoalAttractor(m_u, m_l, alpha_m, k, alpha_psi, k_d)

def dX(t, X):
    x = X[:2].reshape(-1, 1)
    x_dot = X[2:].reshape(-1, 1)
    M, F, _, _, _ = attractor.calc_fabric(x, x_dot, xg)
    a = LA.pinv(M) @ F
    return np.ravel(np.concatenate([x_dot, a]))

sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
print(sol.message)

xi_s, pi_s, d_s, f_s, M_s = [], [], [], [], []
Le_s = []
for i in range(len(sol.t)):
    x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
    x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
    M, F, xi, pi, d = attractor.calc_fabric(x, x_dot, xg)
    M_s.append(M)
    Le_s.append((x_dot.T @ M @ x_dot)[0,0])
    xi_s.append(xi)
    pi_s.append(pi)
    d_s.append(d)
    f_s.append(F)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sol.y[0], sol.y[1], label="line")
ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.grid(); ax.axis('equal'); ax.legend()
fig.savefig("fabric_test.png")

# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 12 ))
# axes[0].plot(sol.t, sol.y[0], label="x")
# axes[0].plot(sol.t, sol.y[1], label="y")
# axes[1].plot(sol.t, sol.y[2], label="dx")
# axes[1].plot(sol.t, sol.y[3], label="dy")
# axes[2].plot(sol.t, xi_s, label="xi")
# axes[2].plot(sol.t, pi_s, label="pi")
# axes[2].plot(sol.t, d_s, label="d")
# axes[2].plot(sol.t, f_s, label="f = pi - xi - d")
# axes[3].plot(sol.t, Le_s, label="geometric Le")
# for ax in axes.ravel():
#     ax.legend()
#     ax.grid()

# fig.savefig("fabric_test_state.png")



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-6, 2)
ax.set_ylim(0, 6)
scale = 100

def update(i):
    ax.cla()
    ax.scatter([xg[0,0]], [xg[1,0]], marker="*", color = "r", label="goal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.plot(sol.y[0][:i], sol.y[1][:i])

    
    eigvals, eigvecs = LA.eig(M_s[i])  # 計量の固有値と固有ベクトルを計算
    if np.any(np.iscomplex(eigvals)) or np.any(eigvals <= 1e-3): # not正定対称．リーマンじゃないのでスキップ
        met_axes_lengths = np.array([0, 0])
        met_angle = 0
    else:  # リーマン計量だから描写
        axes_lengths = 1.0 / np.sqrt(eigvals) * 0.1
        max_len = max(axes_lengths)
        scale = min(2.0 / max_len, 1.0)
        met_axes_lengths = axes_lengths * scale
        met_angle = np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # 楕円の傾き

    #Ellipse(xy = (state_history[0, 12], state_history[0, 13]), width = met[0, 1], height = met[0, 2], angle = met[0, 3], fill = False, color = 'r', lw = 2)
    c = patches.Ellipse(
        xy=(sol.y[0][i], sol.y[1][i]), 
        width = met_axes_lengths[0], height = met_axes_lengths[1], angle = met_angle,
        ec='k', fill=False
    )
    ax.add_patch(c)
    

    ax.set_xlim(-6, 2)
    ax.set_ylim(0, 6)
    ax.grid(); ax.axis('equal'); ax.legend()
    


ani = anm.FuncAnimation(
    fig = fig,
    func = update,
    frames = range(0, len(sol.t), 10),
    interval=50
)
ani.save("fabrictest.gif", writer="pillow")

plt.show()
