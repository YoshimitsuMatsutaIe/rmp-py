import numpy as np
from numpy import linalg as LA
from math import exp, cos, sin, tan
from scipy import integrate
import matplotlib.pyplot as plt
from numba import njit

import attractor_xi_2d
import attractor_xi_3d


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#from typing import Union
import datetime
import time
import os
#from pathlib import Path
import shutil
import json
from typing import Union

from environment import set_point, set_obstacle
import rmp_node
# import rmp_leaf
import tree_constructor
# import mappings
import visualization
from robot_utils import KinematicsAll, get_robot_model, get_cpoint_ids
from planning_ryo import planning



class GoalAttractor:
    def __init__(
        self,
        dim,
        max_speed, gain, sigma_alpha, sigma_gamma,
        wu, wl, alpha, epsilon,
    ):
        self.gain = gain
        self.damp = max_speed / gain
        self.sigma_alpha = sigma_alpha
        self.sigma_gamma = sigma_gamma
        self.wu = wu
        self.wl = wl
        self.alpha = alpha  # ポテンシャルのスケーリング係数
        self.epsilon = epsilon
        
        if dim == 2:
            self.xi_func = attractor_xi_2d.f
        elif dim == 3:
            self.xi_func = attractor_xi_3d.f
        else:
            assert False
    
    
    def calc_rmp_func(self, x, x_dot, xg, xg_dot=None):
        x_norm = LA.norm(x)
        grad_phi = (1-exp(-2*self.alpha*x_norm)) / (1+exp(-2*self.alpha*x_norm)) * x / x_norm
        alpha_x = exp(-x_norm**2 / (2 * self.sigma_alpha**2))
        gamma_x = exp(-x_norm**2 / (2 * self.sigma_gamma**2))
        wx = gamma_x*self.wu + (1 - gamma_x)*self.wl
        
        M = wx*((1-alpha_x) * grad_phi @ grad_phi.T + (alpha_x+self.epsilon) * np.eye(self.dim))

        f = self.M @ (-self.gain*grad_phi - self.damp*x_dot) \
            - self.xi_func(
                x = x,
                x_dot = x_dot,
                sigma_alpha = self.sigma_alpha,
                sigma_gamma = self.sigma_gamma,
                w_u = self.wu,
                w_l = self.wl,
                alpha = self.alpha,
                epsilon = self.epsilon
            )
        
        return M, f



if __name__ == "__main__":
    pass