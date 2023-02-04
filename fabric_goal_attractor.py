import numpy as np
from numba import njit
@njit(cache=True)
def xi_1(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return -2*alpha_m**2*x*x_dot**2*(-m_l + m_u)*np.exp(-alpha_m**2*x**2)
@njit(cache=True)
def grad_psi_1_1(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[0], [0], [0]])
@njit(cache=True)
def pi_1(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[0], [0], [0]])
@njit(cache=True)
def M_1(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return m_l + (-m_l + m_u)*np.exp(-alpha_m**2*x**2)

@njit(cache=True)
def xi_2(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([
        [2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x_dot[0, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x_dot[1, 0]**2], 
        [2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]*x_dot[0, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]*x_dot[1, 0]**2]
        ]) + (np.array([[-4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x_dot[0, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]*x_dot[0, 0]], [-4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x_dot[1, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]*x_dot[1, 0]]])).dot(x_dot)
@njit(cache=True)
def grad_psi_1_2(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[k*(x[0, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))], [k*(x[1, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))]])
@njit(cache=True)
def pi_2(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[k*(-m_l - (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2)))*(x[0, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))], [k*(-m_l - (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2)))*(x[1, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2)))]])
@njit(cache=True)
def M_2(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([
        [m_l + (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2)), 0], 
        [0, m_l + (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2))]
    ])

@njit(cache=True)
def xi_3(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([
        [2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]*x_dot[0, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]*x_dot[1, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]*x_dot[2, 0]**2], 
        [2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]*x_dot[0, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]*x_dot[1, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]*x_dot[2, 0]**2], 
        [2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]*x_dot[0, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]*x_dot[1, 0]**2 + 2*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]*x_dot[2, 0]**2]]) + (np.array([[-4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]*x_dot[0, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]*x_dot[0, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]*x_dot[0, 0]], [-4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]*x_dot[1, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]*x_dot[1, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]*x_dot[1, 0]], [-4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]*x_dot[2, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]*x_dot[2, 0], -4*alpha_m**2*(-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]*x_dot[2, 0]]])).dot(x_dot)
@njit(cache=True)
def grad_psi_1_3(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[k*(x[0, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))], [k*(x[1, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))], [k*(x[2, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))]])
@njit(cache=True)
def pi_3(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[k*(-m_l - (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*(x[0, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[0, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))], [k*(-m_l - (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*(x[1, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[1, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))], [k*(-m_l - (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*(x[2, 0]/np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) - 2*np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))*x[2, 0]/((1 + np.exp(-2*alpha_psi*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))*np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)))]])
@njit(cache=True)
def M_3(x, x_dot, m_u, m_l, alpha_m, k, alpha_psi):
    return np.array([[m_l + (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)), 0, 0], [0, m_l + (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2)), 0], [0, 0, m_l + (-m_l + m_u)*np.exp(-alpha_m**2*(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2))]])