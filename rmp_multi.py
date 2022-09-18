import numpy as np
import numpy.linalg as LA
from math import exp
from numba import njit


import attractor_xi_2d
import attractor_xi_3d


@njit('UniTuple(f8, 2)(f8, f8, f8, f8, f8)', cache=True)
def obs_avoidance_rmp_func(
        x, x_dot,
        gain: float,
        sigma: float,
        rw: float
    ):
    if rw - x > 0:
        w2 = (rw - x)**2 / x
        w2_dot = (-2*(rw-x)*x + (rw-x)) / x**2
    else:
        return (0, 0)
    
    if x_dot < 0:
        u2 = 1 - exp(-x_dot**2 / (2*sigma**2))
        u2_dot = -exp(x_dot**2 / (2*sigma**2)) * (-x_dot/sigma**3)
    else:
        u2 = 0
        u2_dot = 0
    
    delta = u2 + 1/2 * x_dot * u2_dot
    xi = 1/2 * u2 * w2_dot * x_dot**2
    grad_phi = gain * w2 * w2_dot
    
    return (w2 * delta, -grad_phi - xi)



@njit('UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:], f8, f8, f8, f8, f8, f8, f8, f8)', cache=True)
def goal_attractor_rmp_func(
    x, x_dot,
    max_speed: float, gain: float, sigma_alpha: float, sigma_gamma: float,
    wu: float, wl: float, alpha: float, epsilon: float,
):
    damp = max_speed / gain
    x_norm = LA.norm(x)
    grad_phi = (1-exp(-2*alpha*x_norm)) / (1+exp(-2*alpha*x_norm)) * x / x_norm
    alpha_x = exp(-x_norm**2 / (2 * sigma_alpha**2))
    gamma_x = exp(-x_norm**2 / (2 * sigma_gamma**2))
    wx = gamma_x*wu + (1 - gamma_x)*wl
    
    M = wx*((1-alpha_x) * grad_phi @ grad_phi.T + (alpha_x+epsilon) * np.eye(x.shape[0]))

    if x.shape[0] == 2:
        xi = attractor_xi_2d.f(
            x = x,
            x_dot = x_dot,
            sigma_alpha = sigma_alpha,
            sigma_gamma = sigma_gamma,
            w_u = wu,
            w_l = wl,
            alpha = alpha,
            epsilon = epsilon
        )
    elif x.shape[0] == 3:
        xi = attractor_xi_3d.f(
            x = x,
            x_dot = x_dot,
            sigma_alpha = sigma_alpha,
            sigma_gamma = sigma_gamma,
            w_u = wu,
            w_l = wl,
            alpha = alpha,
            epsilon = epsilon
        )
    else:
        assert(False)

    f = M @ (-gain*grad_phi - damp*x_dot) - xi
    
    return (M, f)



@njit('UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8, f8, f8, f8)', cache=True)
def jl_avoidance_rmp_fnc(
    q, q_dot,
    q_max, q_min, q_neutral,
    gamma_p: float,
    gamma_d: float,
    lam: float,
    sigma: float,
):
    dim = q.shape[0]
    xi = np.empty((dim, 1))
    M = np.zeros((dim, dim))
    for i in range(dim):
        alpha_upper = 1 - exp(-max(q_dot[i, 0], 0)**2 / (2*sigma**2))
        alpha_lower = 1 - exp(-min(q_dot[i, 0], 0)**2 / (2*sigma**2))
        s = (q[i,0] - q_min[i,0]) / (q_max[i,0] - q_min[i,0])
        s_dot = 1 / (q_max[i,0] - q_min[i,0])
        d = 4*s*(1-s)
        d_dot = (4 - 8*s) * s_dot
        b =  s*(alpha_upper*d + (1-alpha_upper)) + (1-s)*(alpha_lower*d + (1-alpha_lower))
        b_dot = (s_dot*(alpha_upper*d + (1-alpha_upper)) + s*d_dot) \
            + -s_dot*(alpha_lower * d + (1-alpha_lower)) + (1-s) * d_dot
        a = b**(-2)
        a_dot = -2*b**(-3) * b_dot
        
        xi[i, 0] = 1/2 * a_dot * q_dot[i,0]**2
        M[i,i] = lam * a
    
    f = M @ (gamma_p*(q_neutral - q) - gamma_d*q_dot) - xi
    
    return (M, f)



