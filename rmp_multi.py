import numpy as np
from numpy import pi
import numpy.linalg as LA
from math import exp
from numba import njit, f8, u1
from numba import prange

import attractor_xi_2d
import attractor_xi_3d

import robot_baxter.baxter as baxter
import robot_franka_emika.franka_emika as franka_emika
import robot_sice.sice as sice


@njit('UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:], f8[:,:], f8, f8, f8)', cache=True)
def obs_avoidance_rmp_func(
        x, x_dot, o,
        gain: float,
        sigma: float,
        rw: float
    ):

    s = LA.norm(x - o)
    J = -(x - o) / s
    s_dot = (J @ (x - o))[0,0]
    J_dot = -((x_dot).T - (x-o).T*(1/LA.norm(x-o)*(x-o).T @ (x_dot))) / LA.norm(x-o)**2

    if rw - s > 0:
        w2 = (rw - s)**2 / s
        w2_dot = (-2*(rw-s)*s + (rw-s)) / s**2
    else:
        return (
            np.zeros((x.shape[0], x.shape[0])),
            np.zeros((x.shape[0], 1)),
        )
    
    if s_dot < 0:
        u2 = 1 - exp(-s_dot**2 / (2*sigma**2))
        u2_dot = -exp(s_dot**2 / (2*sigma**2)) * (-s_dot/sigma**3)
    else:
        u2 = 0
        u2_dot = 0
    
    delta = u2 + 1/2 * s_dot * u2_dot
    xi = 1/2 * u2 * w2_dot * s_dot**2
    grad_phi = gain * w2 * w2_dot
    

    m = w2 * delta
    f = -grad_phi - xi

    return (
        m * J.T @ J,
        J.T * (f - (m * J_dot @ x_dot)[0,0])
    )



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



@njit('UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8, f8, f8, f8)', cache=True, parallel=True)
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
    for i in prange(dim):
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



def get_node_ids(robot_name: str):
    if robot_name == "baxter":
        robot_model = baxter
    elif robot_name == "franka_emika":
        robot_model = franka_emika
    elif robot_name == "sice":
        robot_model = sice
    else:
        assert False
    
    ids = []
    for i in range(len(robot_model.CPoint.RS_ALL)):
        for j in range(len(robot_model.CPoint.RS_ALL[i])):
            ids.append((i, j))
    
    return ids


@njit(
    'f8[:,:](f8[:,:], f8[:,:], f8[:,:], ListType(f8[:,:]), DictType(u1, DictType(u1, f8)), u1)',
    cache=True, parallel=True
)
def dX(t, X, c_dim, g, o_s, rmp_param, robot_name: str):
    """solve_ivpにわたすやつ"""
    
    if robot_name == "baxter":
        robot_model = baxter
    elif robot_name == "franka_emika":
        robot_model = franka_emika
    elif robot_name == "sice":
        robot_model = sice
    else:
        assert False
    
    q = X[:c_dim].reshape(-1, 1)
    q_dot = X[c_dim:].reshape(-1, 1)
    
    root_f = np.zeros((c_dim, 1))
    root_M = np.zeros((c_dim, c_dim))
    
    # temp_rmp = jl_avoidance_rmp_fnc(
    #     q, q_dot, 
    #     robot_model.q_max(), robot_model.q_min(), robot_model.q_neutral,
    #     **rmp_param["joint_limit_avoidance"]
    # )
    # root_f = temp_rmp[1]
    # root_M = temp_rmp[0]

    ids = get_node_ids(robot_name)

    for i in prange(len(ids)):
        mapping = robot_model.CPoint(*ids[i])
        x = mapping.phi(q)
        J = mapping.J(q)
        x_dot = J @ q_dot
        J_dot = mapping.J_dot(q, q_dot)

        f = np.zeros((robot_model.CPoint.t_dim, 1))
        M = np.zeros((robot_model.CPoint.t_dim, robot_model.CPoint.t_dim))
        for j in prange(len(o_s)):
            temp_rmp = obs_avoidance_rmp_func(
                x, x_dot, o_s[j], *rmp_param["obstacle_avoidance"]
            )
            f += temp_rmp[1]
            M += temp_rmp[0]

            if ids[i] == robot_model.CPoint.ee_id:
                temp_rmp = goal_attractor_rmp_func(
                    x-g, x_dot, *rmp_param["goal_attractor"]
                )
                f += temp_rmp[1]
                M += temp_rmp[0]
        
        root_f += J.T @ (f - M @ J_dot @ q_dot)
        root_M += J.T @ M @ J


    return LA.pinv(root_M) @ root_f




if __name__ == "__main__":
    pass