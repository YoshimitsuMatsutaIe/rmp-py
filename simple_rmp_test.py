import numpy as np
from numpy import linalg as LA
from scipy import integrate
import sympy as sy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import exp, sin, cos, tan


# rmpの障害物（GDS由来のもの）

xo = np.zeros((2,1))

gain = 5.0
sigma = 1.0
rw = 1


def rmp_obs(x, x_dot):
    x_norm = LA.norm(x)
    if rw - x > 0:
        w2 = (rw - x)**2 / x
        w2_dot = (-2*(rw-x)*x + (rw-x)) / x**2
    else:
        w2 = 0
        w2_dot = 0
    
    if x_dot < 0:
        u2 = 1 - exp(-x_dot**2 / (2*sigma**2))
        u2_dot = -exp(x_dot**2 / (2*sigma**2)) * (-x_dot/sigma**3)
    else:
        u2 = 0
        u2_dot = 0
    print("w2 = {0}, w2_dot = {1}, u2 = {2}, u2_dot = {3}".format(w2, w2_dot, u2, u2_dot))
    delta = u2 + 1/2 * x_dot * u2_dot
    xi = 1/2 * u2 * w2_dot * x_dot**2
    grad_phi = gain * w2 * w2_dot

    M = w2 * delta
    f = -grad_phi - xi
    d = 0
    print("xi = {0}, pi = {1}, m = {2}".format(xi, grad_phi, M))
    return M, f, grad_phi, d, xi


def dX(t, X):
    print("\nt = ", t)
    x = X[:2].reshape(-1, 1)
    x_dot = X[2:].reshape(-1, 1)
    s = LA.norm(x - xo)
    J = (x-xo).T / s
    s_dot = (J @ x_dot)[0,0]
    #print("s = {0}, s_dot = {1}".format(s, s_dot))
    J_dot = (x_dot.T - (x-xo).T*(1/LA.norm(x-xo)*(x-xo).T @ x_dot)) / LA.norm(x-xo)**2
    M, f, _, _, _ = rmp_obs(s, s_dot)
    
    #M = np.minimum(np.maximum(M, - 1e3), 1e3)
    
    pull_M = M * J.T @ J
    pull_f = J.T @ (f - M * J_dot @ x_dot)
    print("pull_M = {0}, pull_f = {1}".format(pull_M, pull_f.T))
    

    
    a = LA.pinv(pull_M) @ pull_f
    return np.ravel(np.concatenate([x_dot, a]))


X0 = np.array([2, 0.5, -0.5, 0.0])
time_interval = 0.01
time_span = 10
tspan = (0, time_span)
teval = np.arange(0, time_span, time_interval)
sol = integrate.solve_ivp(fun=dX, t_span=tspan, y0=X0, t_eval=teval)
print(sol.message)

m_, s_, s_dot_, xi_, pi_, d_,  f_ = [], [], [], [], [], [], []
for i in range(len(sol.t)):
    x = np.array([[sol.y[0][i], sol.y[1][i]]]).T
    x_dot = np.array([[sol.y[2][i], sol.y[3][i]]]).T
    s = LA.norm(x - xo)
    J = (x-xo).T / s
    s_dot = (J @ x_dot)[0,0]
    J_dot = (x_dot.T - (x-xo).T*(1/s*(x-xo).T @ x_dot)) / s**2
    M, f, pi, d, xi = rmp_obs(s, s_dot)
    
    m_.append(M)
    s_.append(s)
    s_dot_.append(s_dot)
    xi_.append(LA.norm(xi))
    pi_.append(LA.norm(pi))
    d_.append(LA.norm(d))
    f_.append(LA.norm(f))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sol.y[0], sol.y[1])
ax.scatter([xo[0,0]], [xo[1,0]], marker="+", color = "r")
c = patches.Circle(xy=(xo[0,0], xo[1,0]), radius=r, ec='k', fill=False)
ax.add_patch(c)
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.grid(); ax.axis('equal')


fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6, 12))
axes[0].plot(sol.t, sol.y[0], label="x")
axes[0].plot(sol.t, sol.y[1], label="y")
axes[1].plot(sol.t, sol.y[2], label="dx")
axes[1].plot(sol.t, sol.y[3], label="dy")
axes[2].plot(sol.t, s_, label="s")
axes[3].plot(sol.t, s_dot_, label="ds")
axes[4].plot(sol.t, m_, label="m")
axes[5].plot(sol.t, xi_, label="xi")
axes[5].plot(sol.t, pi_, label="pi")
axes[5].plot(sol.t, f_, label="f = pi - xi")

for ax in axes.ravel(): ax.legend(); ax.grid()


plt.show()
