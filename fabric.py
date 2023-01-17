import numpy as np
from numpy import linalg as LA
import sympy as sy
from numba import njit


class GoalAttractor:
    def __init__(self, m_u, m_l, alpha_m, k, alpha_psi, k_d):
        self.m_u = m_u
        self.m_l = m_l
        self.alpha_m = alpha_m
        self.k = k
        self.alpha_psi = alpha_psi
        self.k_d = k_d
        self.set_func()
    
    def set_func(self,):
        x = sy.MatrixSymbol('x', 2, 1)
        x_dot = sy.MatrixSymbol('x_dot', 2, 1)
        x_norm = sy.sqrt(x[0,0]**2 + x[1,0]**2)
        m_u, m_l, alpha_m, k, alpha_psi = sy.symbols('m_U, m_l, alpha_m, k, alpha_psi')

        G = (m_u - m_l) * sy.exp(-(alpha_m * x_norm)**2) * sy.eye(2) + m_u * sy.eye(2)
        psi_1 = k * (x_norm + 1/alpha_psi * sy.ln(1 + sy.exp(-2 * alpha_psi * x_norm)))

        L = (x_dot.T * G * x_dot)[0,0]
        M = G

        xi = sy.Matrix([[sy.diff(L, x_dot[0,0]), sy.diff(L, x_dot[1,0])]]).T.jacobian(x) * x_dot \
            - sy.Matrix([[sy.diff(L, x[0,0])], [sy.diff(L, x[1,0])],])
        grad_psi_1 = sy.Matrix([[sy.diff(psi_1, x[0,0]), sy.diff(psi_1, x[1,0])]]).T
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
def ObstacleAvoidance_rmp(x, x_dot, xo, r, k_b, alpha_b):
    
    xxo_norm = LA.norm(x - xo)
    s = xxo_norm / r - 1
    J = 1 / (r * xxo_norm) * (x-xo).T
    s_dot = (J @ x_dot)[0,0]
    J_dot = 1/r * (
        -xxo_norm**(-3/2)*(np.sum(x_dot))*x.T + \
            xxo_norm**(-1/2)*x_dot.T
    )
    
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
    
    def calc_fabric(self, x, x_dot, xo):
        return ObstacleAvoidance_rmp(x, x_dot, xo, self.r, self.k_b, self.alpha_b)



if __name__ == "__main__":
    
    obs = ObstacleAvoidance(1, 1, 1)