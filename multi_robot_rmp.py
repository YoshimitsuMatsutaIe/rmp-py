import numpy as np
from numpy import linalg as LA
import sympy as sy
from numba import njit

@njit
def UnitaryGoalAttractor_a_rmp_func(x, x_dot, xg, xg_dot, gain, wu, wl, sigma, alpha, tol, eta):
    z = x - xg
    z_dot = x_dot - xg_dot
    z_norm = LA.norm(z)
    z_dot_norm = LA.norm(z_dot)
    
    beta = np.exp(- z_norm**2 / 2 / (sigma**2))
    w = (wu - wl) * beta + wl
    s = (1 - np.exp(-2 * alpha * z_norm)) / (1 + np.exp(-2 * alpha * z_norm))

    M = w * np.eye(2)
    
    if z_norm > tol:
        grad_Phi = s / z_norm * w * gain * z
    else:
        grad_Phi = np.zeros((2,1))
    
    Bx_dot = eta * w * z_dot
    grad_w = - beta * (wu - wl) / sigma**2 * z

    xi = -0.5 * (z_dot_norm**2 * grad_w - 2 *
        np.dot(np.dot(z_dot, z_dot.T), grad_w))
    
    F = -grad_Phi - Bx_dot - xi
    return M, F


class UnitaryGoalAttractor_a:
    def __init__(self, gain, wu, wl, sigma, alpha, tol, eta):
        self.gain = gain
        self.wu = wu
        self.wl = wl
        self.sigma = sigma
        self.alpha = alpha
        self.tol = tol
        self.eta = eta #ダンピング
    
    def calc_rmp(self, x, x_dot, xg, xg_dot=np.zeros((2,1))):
        return UnitaryGoalAttractor_a_rmp_func(
            x, x_dot, xg, xg_dot, 
            self.gain, self.wu, self.wl, self.sigma, self.alpha, self.tol, self.eta
        )


@njit
def PairwiseObstacleAvoidance_rmp_func(x, x_dot, xo, Ds, alpha, eta, epsilon):
    xxo_norm = LA.norm(x - xo)
    s = xxo_norm / Ds - 1
    J = 1 / (Ds * xxo_norm) * (x-xo).T
    s_dot = (J @ x_dot)[0,0]
    J_dot = 1/Ds * (
        -xxo_norm**(-3/2)*(np.sum(x_dot))*x.T + \
            xxo_norm**(-1/2)*x_dot.T
    )
    
    if s < 0:
        w = 1e10
        grad_w = 0
    else:
        w = 1.0 / s**4
        grad_w = -4.0 / s**5
    u = epsilon + np.minimum(0, s_dot) * s_dot
    g = w * u

    grad_u = 2 * np.minimum(0, s_dot)
    grad_Phi = alpha * w * grad_w
    xi = 0.5 * s_dot**2 * u * grad_w

    m = g + 0.5 * s_dot * w * grad_u
    #m = np.minimum(np.maximum(m, - 1e5), 1e5)

    Bx_dot = eta * g * s_dot

    f = - grad_Phi - xi - Bx_dot
    #f = np.minimum(np.maximum(f, - 1e10), 1e10)
    
    
    M = m * J.T @ J
    F = J.T * (f + m * (J_dot @ x_dot)[0,0])
    return M, F


class PairwiseObstacleAvoidance:
    """multi-robot-rmpflowのペアワイズ障害物回避"""
    def __init__(self, Ds, alpha, eta, epsilon):
        self.Ds = Ds
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

    def calc_rmp(self, x, x_dot, xo):
        return PairwiseObstacleAvoidance_rmp_func(
            x, x_dot, xo, self.Ds, self.alpha, self.eta, self.epsilon
        )


@njit
def ParwiseDistancePreservation_a_rmp_func(x, x_dot, y, y_dot, d, c, alpha, eta):
    "距離維持rmpを計算"
    
    #print("dis = ", LA.norm(x-y))
    s = LA.norm(x-y) - d
    s_dot = (1/LA.norm(x-y) * (x-y).T @ (x_dot-y_dot))[0,0]
    J = 1 / LA.norm(x-y) * (x-y).T
    J_dot = -LA.norm(x-y)**(-2)*s_dot*(x-y).T + LA.norm(x-y)*(x_dot-y_dot).T
    
    m = c
    grad_phi = alpha * s
    f = -m * grad_phi - m * eta*s_dot
    
    M = m * J.T @ J
    F = J.T * (f + m * (J_dot @ x_dot)[0,0])
    
    return M, F


class ParwiseDistancePreservation_a:
    "rmpflowの距離維持"
    def __init__(self, d, c, alpha, eta,):
        self.d = d #desired distance
        self.c = c #weight
        self.alpha = alpha #attract gain
        self.eta = eta #dampping gaio
    
    def calc_rmp(self, x, x_dot, y, y_dot=np.zeros((2,1))):
        return ParwiseDistancePreservation_a_rmp_func(
            x, x_dot, y, y_dot, self.d, self.c, self.alpha, self.eta
        )


@njit
def NeuralGoalAttractor_rmp_func(x, x_dot, xg, gain, damp, epsilon):
    F = gain * (xg-x) / (LA.norm(xg-x) + epsilon) - damp*x_dot
    M = np.eye(x.shape[0])
    return M, F

class NeuralGoalAttractor:
    def __init__(self, gain, damp, epsilon):
        self.gain = gain
        self.damp = damp
        self.epsilon = epsilon
    
    def calc_rmp(self, x, x_dot, xg, xg_dot=np.zeros((2,1))):
        return NeuralGoalAttractor_rmp_func(x, x_dot, xg, self.gain, self.damp, self.epsilon)


@njit
def NeuralObstacleAvoidancermp_func(x, x_dot, xo, gain, damp, gamma):
    u = (xo - x) / LA.norm(xo - x)
    F = - gain / LA.norm(xo-x) * (damp * (u.T @ x_dot)[0,0] + gamma) * u
    
    s = LA.norm(xo - x)
    w = 1 / (s + 0.2)
    M = w * F @ F.T
    
    return M, F

class NeuralObstacleAvoidance:
    def __init__(self, gain, damp, gamma):
        self.gain = gain
        self.damp = damp
        self.gamma = gamma
    
    def calc_rmp(self, x, x_dot, xo, xo_dot=np.zeros((2,1))):
        return NeuralObstacleAvoidancermp_func(x, x_dot, xo, self.gain, self.damp, self.gamma)




if __name__ == "__main__":
    pass
