import numpy as np
from numpy import linalg as LA
import sympy as sy
from numba import njit


class PairwiseObstacleAvoidance:
    """multi-robot-rmpflowのペアワイズ障害物回避"""
    def __init__(self, Ds, alpha, eta, epsilon):
        self.Ds = Ds
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

    def calc_rmp(self, x, x_dot, xo):
        
        xxo_norm = LA.norm(x - xo)
        s = xxo_norm / self.Ds - 1
        J = 1 / (self.Ds * xxo_norm) * (x-xo).T
        s_dot = (J @ x_dot)[0,0]
        J_dot = 1/self.Ds * (
            -xxo_norm**(-3/2)*(np.sum(x_dot))*x.T + \
                xxo_norm**(-1/2)*x_dot.T
        )
        
        if s < 0:
            w = 1e10
            grad_w = 0
        else:
            w = 1.0 / s**4
            grad_w = -4.0 / s**5
        u = self.epsilon + np.minimum(0, s_dot) * s_dot
        g = w * u

        grad_u = 2 * np.minimum(0, s_dot)
        grad_Phi = self.alpha * w * grad_w
        xi = 0.5 * s_dot**2 * u * grad_w

        m = g + 0.5 * s_dot * w * grad_u
        m = np.minimum(np.maximum(m, - 1e5), 1e5)

        Bx_dot = self.eta * g * s_dot

        f = - grad_Phi - xi - Bx_dot
        f = np.minimum(np.maximum(f, - 1e10), 1e10)
        
        
        M = m * J.T @ J
        F = J.T * (f - m * (J_dot @ x_dot)[0,0])

        return M, F


class ParwiseDistancePreservation_a:
    def __init__(self, d, c, alpha, eta,):
        self.d = d #desired distance
        self.c = c #weight
        self.alpha = alpha #attract gain
        self.eta = eta #dampping gaio
    
    def calc_rmp(self, x, x_dot, y, y_dot=np.zeros((2,1))):
        s = LA.norm(x-y) - self.d
        s_dot = (1/LA.norm(x-y) * (x-y).T @ (x_dot-y_dot))[0,0]
        J = 1 / LA.norm(x-y) * (x-y).T
        J_dot = -LA.norm(x-y)**(-2)*s_dot*(x-y).T + LA.norm(x-y)*(x_dot-y_dot).T
        
        m = self.c
        grad_phi = self.alpha * s
        f = -m * grad_phi - m * self.eta*s_dot
        
        M = m * J.T @ J
        F = J.T * (f - m * (J_dot @ x_dot)[0,0])
        
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
        z = x - xg
        z_dot = x_dot - xg_dot
        z_norm = LA.norm(z)
        beta = np.exp(- z_norm**2 / 2 / (self.sigma**2))
        w = (self.wu - self.wl) * beta + self.wl
        s = (1 - np.exp(-2 * self.alpha * z_norm)) / \
        (1 + np.exp(-2 * self.alpha * z_norm))

        M = w * np.eye(2)
        
        if z_norm > self.tol:
            grad_Phi = s / z_norm * w * z * self.gain
        else:
            grad_Phi = 0
        Bx_dot = self.eta * w * z_dot
        grad_w = - beta * (self.wu - self.wl) / self.sigma**2 * z

        z_dot_norm = LA.norm(z_dot)
        xi = -0.5 * (z_dot_norm ** 2 * grad_w - 2 *
            np.dot(np.dot(z_dot, z_dot.T), grad_w))
        
        F = -grad_Phi - Bx_dot - xi
        
        return M, F
