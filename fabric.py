import numpy as np
from numpy import linalg as LA
import sympy as sy
from numba import njit
from math import sqrt, cos, sin, tan, pi

class GoalAttractor:
    def __init__(self, m_u, m_l, alpha_m, k, alpha_psi, k_d, dim=2):
        self.m_u = m_u
        self.m_l = m_l
        self.alpha_m = alpha_m
        self.k = k
        self.alpha_psi = alpha_psi
        self.k_d = k_d
        self.dim = dim
        self.set_func()
    
    
    def norm(self, x):
        z = 0
        for i in range(self.dim):
            z += x[i,0]**2
        return z**(1/2)
    
    def set_func(self,):
        x = sy.MatrixSymbol('x', self.dim, 1)
        x_dot = sy.MatrixSymbol('x_dot', self.dim, 1)
        x_norm = self.norm(x)
        m_u, m_l, alpha_m, k, alpha_psi = sy.symbols('m_U, m_l, alpha_m, k, alpha_psi')

        G = (m_u - m_l) * sy.exp(-(alpha_m * x_norm)**2) * sy.eye(self.dim) + m_l * sy.eye(self.dim)
        psi_1 = k * (x_norm + 1/alpha_psi * sy.ln(1 + sy.exp(-2 * alpha_psi * x_norm)))

        L = (x_dot.T * G * x_dot)[0,0]
        M = G

        if self.dim == 2:
            xi = sy.Matrix([[sy.diff(L, x_dot[0,0]), sy.diff(L, x_dot[1,0])]]).T.jacobian(x) * x_dot \
                - sy.Matrix([[sy.diff(L, x[0,0])], [sy.diff(L, x[1,0])],])
            grad_psi_1 = sy.Matrix([[sy.diff(psi_1, x[0,0]), sy.diff(psi_1, x[1,0])]]).T
        else:
            xi = sy.Matrix([[sy.diff(L, x_dot[0,0]), sy.diff(L, x_dot[1,0]), sy.diff(L, x_dot[2,0])]]).T.jacobian(x) * x_dot \
                - sy.Matrix([[sy.diff(L, x[0,0])], [sy.diff(L, x[1,0])], [sy.diff(L, x[2,0])]])
            grad_psi_1 = sy.Matrix([[sy.diff(psi_1, x[0,0]), sy.diff(psi_1, x[1,0]), sy.diff(psi_1, x[2,0])]]).T
        pi = -M * grad_psi_1

        self.func_M = sy.lambdify((x, x_dot, m_u, m_l, alpha_m, k, alpha_psi), M, "numpy")
        self.func_xi = sy.lambdify((x, x_dot, m_u, m_l, alpha_m, k, alpha_psi), xi, "numpy")
        self.func_pi = sy.lambdify((x, x_dot, m_u, m_l, alpha_m, k, alpha_psi), pi, "numpy")

    def calc_fabric(self, x, x_dot, xg):
        #print(x)
        z = x - xg
        z_dot = x_dot
        M = self.func_M(z, z_dot, self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi)
        xi = self.func_xi(z, z_dot, self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi)
        pi = self.func_pi(z, z_dot, self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi)
        damp = self.k_d*z_dot
        F = pi - xi - damp
        
        return M, F, xi, pi, damp



@njit
def sgn(x_dot):
    if x_dot < 0:
        return 1
    else:
        return 0


@njit
def task_map2(r, x, x_dot, xo, xo_dot):
    """円形障害物回避"""
    xxo_norm = LA.norm(x - xo)
    s = xxo_norm / r - 1
    J = 1/(r * xxo_norm) * (x-xo).T
    s_dot = (1/(r * xxo_norm) * (x-xo).T @ (x_dot-xo_dot))[0,0]
    s_dot_ = s_dot*r
    J_dot = 1/r * (-xxo_norm**(-2)*s_dot_*(x-xo).T + xxo_norm**(-1)*(x_dot-xo_dot).T)

    return s, s_dot, J, J_dot


@njit
def ObstacleAvoidance_rmp(x, x_dot, xo, xo_dot, r, k_b, alpha_b):
    
    # xxo_norm = LA.norm(x - xo)
    # s = xxo_norm / r - 1
    # J = 1 / (r * xxo_norm) * (x-xo).T
    # s_dot = (J @ x_dot)[0,0]
    # J_dot = 1/r * (-xxo_norm**(-3/2)*(np.sum(x_dot))*x.T + xxo_norm**(-1/2)*x_dot.T)
    s, s_dot, J, J_dot = task_map2(r, x, x_dot, xo, xo_dot)
    
    m = sgn(s_dot) * k_b / s**2
    xi = -2 * s_dot**2 / s**3 * sgn(s_dot)
    pi = -1 * (-4 * alpha_b * s**(-9)) * sgn(s_dot) * s_dot**2 * m
    damp = 0. * s_dot
    f = pi - xi - damp
    
    M = m * J.T @ J
    F = J.T * (f + m * (J_dot @ x_dot)[0,0])
    
    return M, F, m, xi, pi, damp, f


class ObstacleAvoidance:
    def __init__(self, r, k_b, alpha_b):
        self.r = r
        self.k_b = k_b
        self.alpha_b = alpha_b
    
    def calc_fabric(self, x, x_dot, xo, xo_dot):
        return ObstacleAvoidance_rmp(x, x_dot, xo, xo_dot, self.r, self.k_b, self.alpha_b)



@njit
def ParwiseDistancePreservation_rmp(x, x_dot, y, y_dot, d, m_u, m_l, alpha_m, k, alpha_psi, k_d):
    s = LA.norm(x-y) - d
    s_dot = (1/LA.norm(x-y) * (x-y).T @ (x_dot-y_dot))[0,0]
    J = 1 / LA.norm(x-y) * (x-y).T
    J_dot = -LA.norm(x-y)**(-2)*s_dot*(x-y).T + LA.norm(x-y)*(x_dot-y_dot).T
    
    #s, s_dot, J, J_dot = task_map2(d, x, x_dot, y, y_dot)
    
    m =  (m_u - m_l) * np.exp(-(alpha_m * s)**2) + m_l
    xi = -2 * (m_u - m_l) * alpha_m**2 * s**3 * np.exp(-alpha_m*s**2)
    grad_psi = k * (1 - np.exp(-2 * alpha_psi * s)) / (1 + np.exp(-2 * alpha_psi * s))
    
    f =  -m * grad_psi - xi - k_d*s_dot
    
    
    M = m * J.T @ J
    F = J.T * (f + m * (J_dot @ x_dot)[0,0])
    
    return M, F


class ParwiseDistancePreservation:
    """新規性 距離維持"""
    def __init__(self, d, m_u, m_l, alpha_m, k, alpha_psi, k_d):
        self.d = d
        self.m_u = m_u
        self.m_l = m_l
        self.alpha_m = alpha_m
        self.k = k
        self.alpha_psi = alpha_psi
        self.k_d = k_d
    
    def calc_rmp(self, x, x_dot, y, y_dot=None):
        if y_dot is None:
            y_dot = np.zeros(y.shape)
        return ParwiseDistancePreservation_rmp(
            x, x_dot, y, y_dot,
            self.d, self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi, self.k_d
        )



@njit(cache=True)
def angle_taskmap(angle, q1, q2, q3, q1_dot, q2_dot, q3_dot):
    q_x = q1[0,0]; q_y = q1[1,0]
    s_x = q2[0,0]; s_y = q2[1,0]
    t_x = q3[0,0]; t_y = q3[1,0]
    q_x_dot = q1_dot[0,0]; q_y_dot = q1_dot[1,0]
    s_x_dot = q2_dot[0,0]; s_y_dot = q2_dot[1,0]
    t_x_dot = q3_dot[0,0]; t_y_dot = q3_dot[1,0]
    
    g = cos(angle)
    
    x = g - ((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))
    J = np.array([[-(-q_x + s_x)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x + t_x)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_x - 2*s_x - 2*t_x)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)), -(-q_y + s_y)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y + t_y)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_y - 2*s_y - 2*t_y)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))]])
    x_dot = -(-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - ((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))
    J_dot = np.array([[-(-q_x + s_x)*(-3*(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - 3*(-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(5/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x + s_x)*(-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_x + s_x)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x + t_x)*(-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_x + t_x)*(-3*(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - 3*(-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(5/2)) - (-q_x + t_x)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_x_dot + s_x_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x_dot + t_x_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*(4*q_x - 2*s_x - 2*t_x)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*(4*q_x - 2*s_x - 2*t_x)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_x_dot - 2*s_x_dot - 2*t_x_dot)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)), -(-q_y + s_y)*(-3*(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - 3*(-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(5/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y + s_y)*(-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_y + s_y)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y + t_y)*(-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_y + t_y)*(-3*(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - 3*(-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(5/2)) - (-q_y + t_y)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_y_dot + s_y_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y_dot + t_y_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*(4*q_y - 2*s_y - 2*t_y)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*(4*q_y - 2*s_y - 2*t_y)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_y_dot - 2*s_y_dot - 2*t_y_dot)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))]])

    return x, x_dot, J, J_dot


def AnglePreservation_rmp(x, x_dot, y, y_dot, z, z_dot, g, m_u, m_l, alpha_m, k, alpha_psi, k_d):
    s, s_dot, J, J_dot = angle_taskmap(g, x, y, z, x_dot, y_dot, z_dot)
    m =  (m_u - m_l) * np.exp(-(alpha_m * s)**2) + m_l
    xi = -2 * (m_u - m_l) * alpha_m**2 * s**3 * np.exp(-alpha_m*s**2)
    grad_psi = k * (1 - np.exp(-2 * alpha_psi * s)) / (1 + np.exp(-2 * alpha_psi * s))
    
    f =  -m * grad_psi - xi - k_d*s_dot
    
    
    M = m * J.T @ J
    F = J.T * (f + m * (J_dot @ x_dot)[0,0])
    
    return M, F


class AnglePreservation:
    """角度を維持"""
    def __init__(self, m_u, m_l, alpha_m, k, alpha_psi, k_d):
        self.m_u = m_u
        self.m_l = m_l
        self.alpha_m = alpha_m
        self.k = k
        self.alpha_psi = alpha_psi
        self.k_d = k_d
    
    def calc_rmp(self, x, x_dot, y, y_dot, z, z_dot, angle):
        return AnglePreservation_rmp(
            x, x_dot, y, y_dot, z, z_dot,
            angle,
            self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi, self.k_d
        )




if __name__ == "__main__":
    
    #obs = ObstacleAvoidance(1, 1, 1)
    
    q1 = np.random.rand(2, 1)
    q2 = np.random.rand(2, 1)
    q3 = np.random.rand(2, 1)
    q1_dot = np.random.rand(2, 1)
    q2_dot = np.random.rand(2, 1)
    q3_dot = np.random.rand(2, 1)
    
    x, x_dot, J, J_dot = angle_taskmap(
        cos(pi/2), q1, q2, q3, q1_dot, q2_dot, q3_dot
    )
    print(x)
    print(x_dot)
    print(J)
    print(J_dot)