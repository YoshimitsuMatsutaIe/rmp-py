"""真の並列化"""

import numpy as np
import numpy.linalg as LA

# @njit('f8[:,:](f8[:,:], f8[:,:], f8[:,:], f8, f8, f8)', cache=True)
def obs_avoidance_rmp_func(
        x, x_dot, o_s,
        gain: float,
        sigma: float,
        rw: float
    ):

    s_s = LA.norm(x - o_s)
    J_s = -(x - o_s) / s_s
    s_dot_s = (J_s @ (x - o_s))
    J_dot = -((x_dot).T - (x-o).T*(1/LA.norm(x-o)*(x-o).T @ (x_dot))) / LA.norm(x-o)**2

    if rw - s > 0:
        w2 = (rw - s)**2 / s
        w2_dot = (-2*(rw-s)*s + (rw-s)) / s**2
    else:
        return np.zeros((x.shape[0], x.shape[0]+1))
    
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

    RMP = np.empty((x.shape[0], x.shape[0]+1))
    RMP[:, x.shape[0]:] = J.T * (f - (m * J_dot @ x_dot)[0,0])
    RMP[:, :x.shape[0]] = m * J.T @ J

    return RMP



if __name__ == "__main__":
    pass