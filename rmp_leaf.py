import numpy as np
from numpy import linalg as LA
from math import exp
from typing import Union

import mappings
import rmp_node
import attractor_xi_2d
import attractor_xi_3d


class LeafBase(rmp_node.Node):
    def __init__(
        self,
        name: str,
        dim: int,
        parent: Union[rmp_node.Node, None],
        mappings: mappings.Identity,
        parent_dim: Union[int, None]=None,
    ):
        super().__init__(
            name = name,
            dim = dim,
            parent = parent,
            mappings = mappings,
            parent_dim = parent_dim
        )
        self.children = []
    
    
    def print_all_state(self):
        self.print_state()
    
    def add_child(self,):
        pass
    
    def pushforward(self,):
        pass
    
    
    def pullback(self):
        self.calc_rmp_func()
        assert self.parent is not None, "pulled at " + self.name + ", error"
        self.parent.f += self.J.T @ (self.f - self.M @ self.J_dot @ self.parent.x_dot)
        self.parent.M += self.J.T @ self.M @ self.J
    
    
    def calc_rmp_func(self,):
        pass
    
    
    def set_state(self, x, x_dot):
        self.x = x
        self.x_dot = x_dot



class GoalAttractor(LeafBase):
    def __init__(
        self, name: str, parent: rmp_node.Node,
        dim: int,
        calc_mappings: mappings.Identity,
        max_speed: float, gain: float, sigma_alpha: float, sigma_gamma: float,
        wu: float, wl: float, alpha: float, epsilon: float,
    ):
        self.gain = gain
        self.damp = max_speed / gain
        self.sigma_alpha = sigma_alpha
        self.sigma_gamma = sigma_gamma
        self.wu = wu
        self.wl = wl
        self.alpha = alpha  # ポテンシャルのスケーリング係数
        self.epsilon = epsilon
        
        assert dim == 2 or dim == 3, "must dim = 2 or 3"
        if dim == 2:
            self.xi_func = attractor_xi_2d.f
        elif dim == 3:
            self.xi_func = attractor_xi_3d.f
        else:
            assert False
        
        super().__init__(name, dim, parent, calc_mappings,)
    
    
    def calc_rmp_func(self,):
        #print("error = ", self.x.T)
        self.M = self.__inertia_matrix()
        self.f = self.__force()
    
    
    def __grad_phi(self,):
        x_norm = LA.norm(self.x)
        return (1-exp(-2*self.alpha*x_norm)) / (1+exp(-2*self.alpha*x_norm)) * self.x / x_norm
    
    
    def __inertia_matrix(self,):
        x_norm = LA.norm(self.x)
        alpha_x = exp(-x_norm**2 / (2 * self.sigma_alpha**2))
        gamma_x = exp(-x_norm**2 / (2 * self.sigma_gamma**2))
        wx = gamma_x*self.wu + (1 - gamma_x)*self.wl
        
        grad = self.__grad_phi()
        return wx*((1-alpha_x) * grad @ grad.T + (alpha_x+self.epsilon) * np.eye(self.dim))
    
    
    def __force(self,):
        xi = self.xi_func(
            x = self.x,
            x_dot = self.x_dot,
            sigma_alpha = self.sigma_alpha,
            sigma_gamma = self.sigma_gamma,
            w_u = self.wu,
            w_l = self.wl,
            alpha = self.alpha,
            epsilon = self.epsilon
        )
        return self.M @ (-self.gain*self.__grad_phi() - self.damp*self.x_dot) - xi



class ObstacleAvoidance(LeafBase):
    def __init__(
        self, name, parent, calc_mappings,
        scale_rep: float,
        scale_damp: float,
        gain: float,
        sigma: float,
        rw: float
    ):
        self.scale_rep = scale_rep
        self.scale_damp = scale_damp
        self.gain = gain
        self.sigma = sigma
        self.rw = rw
    
        super().__init__(name, 1, parent, calc_mappings,)
    
    
    def calc_rmp_func(self,):
        s = self.x
        s_dot = self.x_dot[0,0]
        
        if self.rw - s > 0:
            w2 = (self.rw - s)**2 / s
            w2_dot = (-2*(self.rw-s)*s + (self.rw-s)) / s**2
        else:
            self.M[0,0] = 0
            self.f[0,0] = 0
            return
        
        if s_dot < 0:
            u2 = 1 - exp(-s_dot**2 / (2*self.sigma**2))
            u2_dot = -exp(s_dot**2 / (2*self.sigma**2)) * (-s_dot/self.sigma**3)
        else:
            u2 = 0
            u2_dot = 0
        
        delta = u2 + 1/2 * s_dot * u2_dot
        xi = 1/2 * u2 * w2_dot * s_dot**2
        grad_phi = self.gain * w2 * w2_dot
        
        self.M[0,0] = w2 * delta
        self.f[0,0] = -grad_phi - xi



class JointLimitAvoidance(LeafBase):
    def __init__(
        self, name, parent: Union[rmp_node.Node, None], calc_mappings,
        gamma_p: float,
        gamma_d: float,
        lam: float,
        sigma: float,
        q_max,
        q_min,
        q_neutral,
        parent_dim: Union[int, None]=None
    ):
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.lam = lam
        self.sigma = sigma
        self.q_max = q_max
        self.q_min = q_min
        self.q_neutral = q_neutral
        
        if parent is not None:
            super().__init__(name, parent.dim, parent, calc_mappings)
        else:
            assert parent_dim is not None
            super().__init__(name, parent_dim, parent, calc_mappings)
    
    
    def pullback(self):
        if not self.isMulti:
            super().pullback()
        else:
            self.calc_rmp_func()
    
    def calc_rmp_func(self,):
        xi = np.empty((self.dim, 1))
        
        for i in range(self.dim):
            alpha_upper = 1 - exp(-max(self.x_dot[i, 0], 0)**2 / (2*self.sigma**2))
            alpha_lower = 1 - exp(-min(self.x_dot[i, 0], 0)**2 / (2*self.sigma**2))
            s = (self.x[i,0] - self.q_min[i,0]) / (self.q_max[i,0] - self.q_min[i,0])
            s_dot = 1 / (self.q_max[i,0] - self.q_min[i,0])
            d = 4*s*(1-s)
            d_dot = (4 - 8*s) * s_dot
            b =  s*(alpha_upper*d + (1-alpha_upper)) + (1-s)*(alpha_lower*d + (1-alpha_lower))
            b_dot = (s_dot*(alpha_upper*d + (1-alpha_upper)) + s*d_dot) \
                + -s_dot*(alpha_lower * d + (1-alpha_lower)) + (1-s) * d_dot
            a = b**(-2)
            a_dot = -2*b**(-3) * b_dot
            
            xi[i, 0] = 1/2 * a_dot * self.x_dot[i,0]**2
            self.M[i,i] = self.lam * a
        
        self. f = self.M @ (self.gamma_p*(self.q_neutral - self.x) - self.gamma_d*self.x_dot) - xi


# class GoalAttractor_1(rmp_tree.LeafBase):
#     """曲率なし"""
#     def __init__(
#         self, name, parent, dim, calc_mappings,
#         max_speed, gain, a_damp_r, sigma_W, sigma_H, A_damp_r
#     ):
#         super().__init__(name, dim, parent, calc_mappings)
#         self.gain = gain
#         self.damp = gain / max_speed
#         self.a_damp_r = a_damp_r
#         self.sigma_W = sigma_W
#         self.sigma_H = sigma_H
#         self.A_damp_r = A_damp_r
    
    
#     def calc_rmp_func(self,):
#         a = self.__acceleration()
#         self.M = self.__inertia_matrix(a)
#         self.f = self.M @ a
    
#     def __soft_normal(self, v, alpha):
#         """ソフト正規化関数"""
#         v_norm = LA.norm(v)
#         softmax = v_norm + 1 / alpha * log(1 + exp(-2 * alpha * v_norm))
#         return v / softmax

#     def __metric_stretch(self, v, alpha):
#         """空間を一方向に伸ばす計量"""
#         xi = self.__soft_normal(v, alpha)
#         return xi @ xi.T

#     def __basic_metric_H(self, f, alpha, beta):
#         """基本の計量"""
#         f_norm = LA.norm(f)
#         f_softmax = f_norm + 1 / alpha * log(1 + exp(-2 * alpha * f_norm))
#         s = f / f_softmax
#         return beta * s @ s.T + (1 - beta) * np.eye(3)
    
    
#     def __acceleration(self,):
#         return -self.gain * self.__soft_normal(self.x, self.a_damp_r) - self.damp * self.x_dot

#     def __inertia_matrix(self, acceleration):
#         d = LA.norm(self.x)
#         weight = exp(- d/ self.sigma_W)
#         beta_attract = 1 - exp(-1 / 2 * (d / self.sigma_H) ** 2)
        
#         return weight * self.__basic_metric_H(acceleration, self.A_damp_r, beta_attract)




if __name__ == "__main__":
    pass