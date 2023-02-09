import numpy as np
from numpy import linalg as LA
import sympy as sy
from numba import njit
from math import sqrt, cos, sin, tan, pi, exp
import fabric_goal_attractor

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
        # #print("goal!")
        # x = sy.MatrixSymbol('x', self.dim, 1)
        # x_dot = sy.MatrixSymbol('x_dot', self.dim, 1)
        # x_norm = self.norm(x)
        # m_u, m_l, alpha_m, k, alpha_psi = sy.symbols('m_U, m_l, alpha_m, k, alpha_psi')

        # G = (m_u - m_l) * sy.exp(-(alpha_m * x_norm)**2) * sy.eye(self.dim) + m_l * sy.eye(self.dim)
        # psi_1 = k * (x_norm + 1/alpha_psi * sy.ln(1 + sy.exp(-2 * alpha_psi * x_norm)))

        # L = (x_dot.T * G * x_dot)[0,0]
        # M = G

        # if self.dim == 2:
        #     xi = sy.Matrix([[sy.diff(L, x_dot[0,0]), sy.diff(L, x_dot[1,0])]]).T.jacobian(x) * x_dot \
        #         - sy.Matrix([[sy.diff(L, x[0,0])], [sy.diff(L, x[1,0])],])
        #     grad_psi_1 = sy.Matrix([[sy.diff(psi_1, x[0,0]), sy.diff(psi_1, x[1,0])]]).T
        # else:
        #     xi = sy.Matrix([[sy.diff(L, x_dot[0,0]), sy.diff(L, x_dot[1,0]), sy.diff(L, x_dot[2,0])]]).T.jacobian(x) * x_dot \
        #         - sy.Matrix([[sy.diff(L, x[0,0])], [sy.diff(L, x[1,0])], [sy.diff(L, x[2,0])]])
        #     grad_psi_1 = sy.Matrix([[sy.diff(psi_1, x[0,0]), sy.diff(psi_1, x[1,0]), sy.diff(psi_1, x[2,0])]]).T
        # pi = -M * grad_psi_1

        # self.func_M = sy.lambdify((x, x_dot, m_u, m_l, alpha_m, k, alpha_psi), M, "numpy")
        # self.func_xi = sy.lambdify((x, x_dot, m_u, m_l, alpha_m, k, alpha_psi), xi, "numpy")
        # self.func_pi = sy.lambdify((x, x_dot, m_u, m_l, alpha_m, k, alpha_psi), pi, "numpy")
        # #print("done!")
        if self.dim == 1:
            self.func_M = fabric_goal_attractor.M_1
            self.func_xi = fabric_goal_attractor.xi_1
            self.func_pi = fabric_goal_attractor.pi_1
        elif self.dim == 2:
            self.func_M = fabric_goal_attractor.M_2
            self.func_xi = fabric_goal_attractor.xi_2
            self.func_pi = fabric_goal_attractor.pi_2
        elif self.dim == 3:
            self.func_M = fabric_goal_attractor.M_3
            self.func_xi = fabric_goal_attractor.xi_3
            self.func_pi = fabric_goal_attractor.pi_3
        else:
            assert False
        return

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



@njit(cache=True)
def sgn(x_dot):
    if x_dot < 0:
        return 1
    else:
        return 0


@njit(cache=True)
def task_map2(r, x, x_dot, xo, xo_dot):
    """円形障害物回避"""
    xxo_norm = LA.norm(x - xo)
    s = xxo_norm / r - 1
    J = 1/(r * xxo_norm) * (x-xo).T
    s_dot = (1/(r * xxo_norm) * (x-xo).T @ (x_dot-xo_dot))[0,0]
    s_dot_ = s_dot*r
    J_dot = 1/r * (-xxo_norm**(-2)*s_dot_*(x-xo).T + xxo_norm**(-1)*(x_dot-xo_dot).T)

    return s, s_dot, J, J_dot


@njit(cache=True)
def ObstacleAvoidance_func(x, x_dot, xo, xo_dot, r, k_b, alpha_b):
    
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
        return ObstacleAvoidance_func(x, x_dot, xo, xo_dot, self.r, self.k_b, self.alpha_b)


@njit(cache=True)
def ObstacleAvoidance2_func(x, x_dot, xo, xo_dot, r, ag, ap, k):
    s, s_dot, J, J_dot = task_map2(r, x, x_dot, xo, xo_dot)
    m = ag * np.exp(-ag*s) * sgn(s_dot)
    xi = -1/2 * s_dot**2 * ag**2 * np.exp(-ag*s) * sgn(s_dot)
    pi = k * ag * ap**2 * np.exp(-(ag+ap)*s) * sgn(s_dot)
    damp = 0. * s_dot
    
    f = pi - xi - damp
    
    M = m * J.T @ J
    F = J.T * (f + m * (J_dot @ x_dot)[0,0])
    
    return M, F, m, xi, pi, damp, f


class ObstacleAvoidance2:
    """proposed"""
    def __init__(self, r, ag, ap, k):
        self.r = r
        self.k = k
        self.ag = ag
        self.ap = ap
    
    def calc_fabric(self, x, x_dot, xo, xo_dot):
        return ObstacleAvoidance2_func(x, x_dot, xo, xo_dot, self.r, self.ag, self.ap, self.k)



@njit(cache=True)
def ParwiseDistancePreservation_func(x, x_dot, y, y_dot, d, m_u, m_l, alpha_m, k, alpha_psi, k_d):
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
    def __init__(self, m_u, m_l, alpha_m, k, alpha_psi, k_d):
        self.m_u = m_u
        self.m_l = m_l
        self.alpha_m = alpha_m
        self.k = k
        self.alpha_psi = alpha_psi
        self.k_d = k_d
    
    def calc_rmp(self, d, x, x_dot, y, y_dot=None):
        if y_dot is None:
            y_dot = np.zeros(y.shape)
        return ParwiseDistancePreservation_func(
            x, x_dot, y, y_dot,
            d, self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi, self.k_d
        )



# #@njit(cache=True)
# def angle_taskmap(angle, q1, q2, q3, q1_dot, q2_dot, q3_dot):
#     q_x = q1[0,0]; q_y = q1[1,0]
#     s_x = q2[0,0]; s_y = q2[1,0]
#     t_x = q3[0,0]; t_y = q3[1,0]
#     q_x_dot = q1_dot[0,0]; q_y_dot = q1_dot[1,0]
#     s_x_dot = q2_dot[0,0]; s_y_dot = q2_dot[1,0]
#     t_x_dot = q3_dot[0,0]; t_y_dot = q3_dot[1,0]
    
#     g = cos(angle)
    
#     x = g - ((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))
#     J = np.array([[-(-q_x + s_x)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x + t_x)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_x - 2*s_x - 2*t_x)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)), -(-q_y + s_y)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y + t_y)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_y - 2*s_y - 2*t_y)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))]])
#     x_dot = -(-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - ((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))
#     J_dot = np.array([[-(-q_x + s_x)*(-3*(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - 3*(-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(5/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x + s_x)*(-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_x + s_x)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x + t_x)*(-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_x + t_x)*(-3*(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - 3*(-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(5/2)) - (-q_x + t_x)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_x_dot + s_x_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_x_dot + t_x_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*(4*q_x - 2*s_x - 2*t_x)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*(4*q_x - 2*s_x - 2*t_x)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_x_dot - 2*s_x_dot - 2*t_x_dot)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)), -(-q_y + s_y)*(-3*(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - 3*(-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(5/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y + s_y)*(-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_y + s_y)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y + t_y)*(-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_y + t_y)*(-3*(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - 3*(-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(5/2)) - (-q_y + t_y)*((-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot) + (-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot) + (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot) + (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot) - (s_x - t_x)*(2*s_x_dot - 2*t_x_dot) - (s_y - t_y)*(2*s_y_dot - 2*t_y_dot))/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-q_y_dot + s_y_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-q_y_dot + t_y_dot)*((-q_x + s_x)**2 + (-q_x + t_x)**2 + (-q_y + s_y)**2 + (-q_y + t_y)**2 - (s_x - t_x)**2 - (s_y - t_y)**2)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (-(-q_x + s_x)*(-2*q_x_dot + 2*s_x_dot)/2 - (-q_y + s_y)*(-2*q_y_dot + 2*s_y_dot)/2)*(4*q_y - 2*s_y - 2*t_y)/(2*((-q_x + s_x)**2 + (-q_y + s_y)**2)**(3/2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2)) - (-(-q_x + t_x)*(-2*q_x_dot + 2*t_x_dot)/2 - (-q_y + t_y)*(-2*q_y_dot + 2*t_y_dot)/2)*(4*q_y - 2*s_y - 2*t_y)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*((-q_x + t_x)**2 + (-q_y + t_y)**2)**(3/2)) - (4*q_y_dot - 2*s_y_dot - 2*t_y_dot)/(2*sqrt((-q_x + s_x)**2 + (-q_y + s_y)**2)*sqrt((-q_x + t_x)**2 + (-q_y + t_y)**2))]])

#     #print("x = {0}, x_dot = {1}".format(x, x_dot))
#     #print("J = ", J)
#     #print("")
#     return x, x_dot, J, J_dot


# #@njit(cache=True)
# def AnglePreservation_func(x, x_dot, y, y_dot, z, z_dot, g, m_u, m_l, alpha_m, k, alpha_psi, k_d):
#     s, s_dot, J, J_dot = angle_taskmap(g, x, y, z, x_dot, y_dot, z_dot)
#     #m =  (m_u - m_l) * np.exp(-(alpha_m * s)**2) + m_l
#     m = 1
#     xi = -2 * (m_u - m_l) * alpha_m**2 * s**3 * np.exp(-alpha_m*s**2)
#     grad_psi = k * (1 - np.exp(-2 * alpha_psi * s)) / (1 + np.exp(-2 * alpha_psi * s))
    
#     f =  -m * grad_psi - xi - k_d*s_dot
    
    
#     M = m * J.T @ J
#     #F = J.T * (f + m * (J_dot @ x_dot)[0,0])
#     F = J.T * f
#     #print("F = ", F.T)
#     return M, F


# class AnglePreservation:
#     """角度を維持"""
#     def __init__(self, m_u, m_l, alpha_m, k, alpha_psi, k_d):
#         self.m_u = m_u
#         self.m_l = m_l
#         self.alpha_m = alpha_m
#         self.k = k
#         self.alpha_psi = alpha_psi
#         self.k_d = k_d
    
#     def calc_rmp(self, x, x_dot, y, y_dot, z, z_dot, angle):
#         return AnglePreservation_func(
#             x, x_dot, y, y_dot, z, z_dot,
#             angle,
#             self.m_u, self.m_l, self.alpha_m, self.k, self.alpha_psi, self.k_d
#         )



@njit(cache=True)
def SpaceLimitAvoidance_func(q, q_dot, q_min, q_max, r, sigma, a, k):
    
    dim = q.shape[0]
    xi = np.zeros((dim, 1))
    grad = np.zeros((dim, 1))
    M = np.zeros((dim, dim))
    for i in range(dim):
        x = q[i,0]; x_dot = q_dot[i,0]
        xl = q_min[i,0]; xu = q_max[i,0]
        su = sgn(-x_dot); sl = sgn(x_dot)
        xi[i,0] = x_dot**2*(a**2*sl*(1 - exp(sigma*(-r + x - xl)))*exp(-a*(x - xl)) + a**2*su*(exp(sigma*(r + x - xu)) + 1)*exp(a*(x - xu)) + a*sigma*sl*exp(-a*(x - xl))*exp(sigma*(-r + x - xl)) + a*sigma*su*exp(a*(x - xu))*exp(sigma*(r + x - xu)))
        
        bu = a * exp(a*(x - xu)) * (1 + exp(-sigma*(x-(xu-r)))**(-1)) *  su
        bl = -a * exp(-a*(x - xl)) * (1 - exp(-sigma*(x-(xl+r)))**(-1)) * sl
        g = bu + bl
        M[i,i] = g
        
        grad[i,0] = k * ((x-xu)**(-3) + (x-xl)**(-3))
    
    pi = -M @ grad
    F = pi - xi
    if LA.norm(F) > 10:
        print("F = ", F.T)
    return M, F


class SpaceLimitAvoidance:
    def __init__(self, r, sigma, a, k, x_max, x_min, y_max, y_min, z_max=None, z_min=None):
        if z_max is None:
            self.q_max = np.array([[x_max, y_max]]).T
            self.q_min = np.array([[x_min, y_min]]).T
        else:
            self.q_max = np.array([[x_max, y_max, z_max]]).T
            self.q_min = np.array([[x_min, y_min, z_min]]).T
        self.r = r
        self.sigma = sigma
        self.a = a
        self.k = k

    def calc_rmp(self, x, x_dot):
        return SpaceLimitAvoidance_func(
            x, x_dot, 
            self.q_min, self.q_max, self.r, self.sigma, self.a, self.k
        )



def SpaceLimitAvoidance_func_2(x, x_dot, gamma_p, gamma_d, lam, sigma, q_max, q_min, q_neutral):
    dim = x.shape[0]
    xi = np.zeros(x.shape)
    M = np.zeros((dim, dim))

    for i in range(dim):
        alpha_upper = 1 - exp(-max(x_dot[i, 0], 0)**2 / (2*sigma**2))
        alpha_lower = 1 - exp(-min(x_dot[i, 0], 0)**2 / (2*sigma**2))
        s = (x[i,0] - q_min[i,0]) / (q_max[i,0] - q_min[i,0])
        s_dot = 1 / (q_max[i,0] - q_min[i,0])
        d = 4*s*(1-s)
        d_dot = (4 - 8*s) * s_dot
        b =  s*(alpha_upper*d + (1-alpha_upper)) + (1-s)*(alpha_lower*d + (1-alpha_lower))
        b_dot = (s_dot*(alpha_upper*d + (1-alpha_upper)) + s*d_dot) \
            + -s_dot*(alpha_lower * d + (1-alpha_lower)) + (1-s) * d_dot
        a = b**(-2)
        a_dot = -2*b**(-3) * b_dot
        
        xi[i, 0] = 1/2 * a_dot * x_dot[i,0]**2
        M[i,i] = lam * a
    
    F = M @ (gamma_p*(q_neutral - x) - gamma_d*x_dot) - xi

    return M, F


class SpaceLimitAvoidance_2:
    def __init__(
        self,
        gamma_p, gamma_d, lam, sigma,
        x_max, x_min, x_0,
        y_max, y_min, y_0,
        z_max=None, z_min=None, z_0=None
    ):
        if z_max is None:
            self.q_max = np.array([[x_max, y_max]]).T
            self.q_min = np.array([[x_min, y_min]]).T
            self.q_neutral = np.array([[x_0, y_0,]]).T
        else:
            self.q_max = np.array([[x_max, y_max, z_max]]).T
            self.q_min = np.array([[x_min, y_min, z_min]]).T
            self.q_neutral = np.array([[x_0, y_0, z_0]]).T
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.lam = lam
        self.sigma = sigma

    def calc_rmp(self, x, x_dot):
        return SpaceLimitAvoidance_func_2(
            x, x_dot, 
            self.gamma_p, self.gamma_d, self.lam, self.sigma, 
            self.q_max, self.q_min, self.q_neutral
        )



if __name__ == "__main__":
    pass
    #obs = ObstacleAvoidance(1, 1, 1)
    
    # q1 = np.random.rand(2, 1)
    # q2 = np.random.rand(2, 1)
    # q3 = np.random.rand(2, 1)
    # q1_dot = np.random.rand(2, 1)
    # q2_dot = np.random.rand(2, 1)
    # q3_dot = np.random.rand(2, 1)
    
    # x, x_dot, J, J_dot = angle_taskmap(
    #     cos(pi/2), q1, q2, q3, q1_dot, q2_dot, q3_dot
    # )
    # print(x)
    # print(x_dot)
    # print(J)
    # print(J_dot)