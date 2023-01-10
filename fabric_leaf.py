
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

from math import exp
from typing import Union, Tuple
from numba import njit

import mappings
import rmp_node
from rmp_leaf import LeafBase



class GoalAttractor(LeafBase):
    def __init__(
        self, name,
        dim,
        calc_mappings: mappings.Identity,
        max_speed, gain, sigma_alpha, sigma_gamma,
        mu, ml, alpha_m, epsilon,
        k, alpha_psi,
        parent_dim
    ):
        self.gain = gain
        self.damp = max_speed / gain
        self.sigma_alpha = sigma_alpha
        self.sigma_gamma = sigma_gamma
        self.mu = mu
        self.ml = ml
        self.alpha_m = alpha_m
        self.k = k
        self.alpha_psi = alpha_psi
        self.epsilon = epsilon
        

        super().__init__(name, dim, calc_mappings, parent_dim)
    
    
    def calc_rmp_func(self):
        
        x_norm = LA.norm(self.x)
        
        G = ((self.mu - self.ml) * exp(-self.alpha_m*x_norm**2) + self.ml) * np.eye(self.dim)
        self.M = G
        
        grad_psi = self.k * \
            (1 - exp(-2*self.alpha_psi*x_norm)) / (1 + exp(-2*self.alpha_psi*x_norm)) * \
                self.x / x_norm
        
        
        