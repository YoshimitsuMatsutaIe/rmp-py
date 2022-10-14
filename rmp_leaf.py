
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

from math import exp
from typing import Union, Tuple
from numba import njit

import mappings
import rmp_node
import attractor_xi_2d
import attractor_xi_3d


class LeafBase(rmp_node.Node):
    def __init__(
        self,
        name: str,
        dim: int,
        mappings: mappings.Identity,
        parent_dim: Union[int, None]=None,
    ):
        super().__init__(
            name = name,
            dim = dim,
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
        self, name: str,
        dim: int,
        calc_mappings: mappings.Identity,
        max_speed: float, gain: float, sigma_alpha: float, sigma_gamma: float,
        wu: float, wl: float, alpha: float, epsilon: float,
        parent_dim: Union[int, None]=None
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
        
        super().__init__(name, dim, calc_mappings, parent_dim)
    
    
    def calc_rmp_func(self):

        x_norm = LA.norm(self.x)
        grad_phi = (1-exp(-2*self.alpha*x_norm)) / (1+exp(-2*self.alpha*x_norm)) * self.x / x_norm
        alpha_x = exp(-x_norm**2 / (2 * self.sigma_alpha**2))
        gamma_x = exp(-x_norm**2 / (2 * self.sigma_gamma**2))
        wx = gamma_x*self.wu + (1 - gamma_x)*self.wl
        
        self.M = wx*((1-alpha_x) * grad_phi @ grad_phi.T + (alpha_x+self.epsilon) * np.eye(self.dim))

        self.f = self.M @ (-self.gain*grad_phi - self.damp*self.x_dot) \
            - self.xi_func(
                x = self.x,
                x_dot = self.x_dot,
                sigma_alpha = self.sigma_alpha,
                sigma_gamma = self.sigma_gamma,
                w_u = self.wu,
                w_l = self.wl,
                alpha = self.alpha,
                epsilon = self.epsilon
            )




class ObstacleAvoidance(LeafBase):
    def __init__(
        self, name, calc_mappings,
        gain: float,
        sigma: float,
        rw: float,
        parent_dim: Union[int, None]=None
    ):
        self.gain = gain
        self.sigma = sigma
        self.rw = rw
    
        self.name = name
        self.dim= 1
        self.parent = None
        self.mappings = calc_mappings
        self.children = []
        self.isMulti = True
        

        self.x = 0
        self.x_dot = 0
        self.f = 0
        self.M = 0
        if parent_dim is not None:
            self.J = np.empty((1, parent_dim))
            self.J_dot = np.empty((1, parent_dim))
    
    
    def calc_rmp_func(self,):
        if self.rw - self.x > 0:
            w2 = (self.rw - self.x)**2 / self.x
            w2_dot = (-2*(self.rw-self.x)*self.x + (self.rw-self.x)) / self.x**2
        else:
            self.M = 0
            self.f = 0
            return
        
        if self.x_dot < 0:
            u2 = 1 - exp(-self.x_dot**2 / (2*self.sigma**2))
            u2_dot = -exp(self.x_dot**2 / (2*self.sigma**2)) * (-self.x_dot/self.sigma**3)
        else:
            u2 = 0
            u2_dot = 0
        
        delta = u2 + 1/2 * self.x_dot * u2_dot
        xi = 1/2 * u2 * w2_dot * self.x_dot**2
        grad_phi = self.gain * w2 * w2_dot
        
        self.M = w2 * delta
        self.f = -grad_phi - xi


    def pullback(self):
        self.calc_rmp_func()
        assert self.parent is not None, "pulled at " + self.name + ", error"
        self.parent.f += self.J.T * (self.f - (self.M * self.J_dot @ self.parent.x_dot)[0,0])
        self.parent.M += self.M * self.J.T @ self.J




class ObstacleAvoidanceMulti(LeafBase):
    def __init__(
        self, name, calc_mappings: mappings.Identity,
        dim: int,
        o_s: npt.NDArray,
        gain: float,
        sigma: float,
        rw: float,
        parent_dim: Union[int, None]=None
    ):
        self.gain = gain
        self.sigma = sigma
        self.rw = rw
        self.o_s = o_s
    
        super().__init__(name, dim, calc_mappings, parent_dim)

        self.isMulti = True
    
    
    def __get_near_obs_id(self):
        dis = self.rw - LA.norm(self.x - self.o_s, axis=0)
        return np.where(dis > 0)[0]
    
    
    def __calc_rmp_func(self, s: float, s_dot: float) -> Tuple[float, float]:

        w2 = (self.rw - s)**2 / s
        w2_dot = (-2*(self.rw-s)*s + (self.rw-s)) / s**2

        if s_dot < 0:
            assert abs(s_dot) < 1e+3, "s = {0}, s_dot = {1}".format(s, s_dot)
            u2 = 1 - exp(-s_dot**2 / (2*self.sigma**2))
            u2_dot = -exp(s_dot**2 / (2*self.sigma**2)) * (-s_dot/self.sigma**3)
        else:
            u2 = 0
            u2_dot = 0
        
        delta = u2 + 1/2 * s_dot * u2_dot
        xi = 1/2 * u2 * w2_dot * s_dot**2
        grad_phi = self.gain * w2 * w2_dot
        
        m = w2 * delta
        f = -grad_phi - xi
        
        return m, f
    
    
    def calc_rmp_func(self,):
        
        obs_ids = self.__get_near_obs_id()
        
        self.f.fill(0)
        self.M.fill(0)
        
        if len(obs_ids) == 0:
            return
        else:
            for id in obs_ids:
                z = self.x - self.o_s[:, id:id+1]
                s = LA.norm(z)
                #print(s)
                J = -z.T / s
                s_dot = (J @ self.x_dot)[0,0]
                J_dot = -(self.x_dot.T - z.T*(1/LA.norm(z)*z.T @ self.x_dot)) / s**2
                
                m, f = self.__calc_rmp_func(s, s_dot)
                
                self.M += m * J.T @ J
                self.f += J.T * (f - (m * J_dot @ self.x_dot)[0,0])



class JointLimitAvoidance(LeafBase):
    def __init__(
        self, name, calc_mappings,
        gamma_p: float,
        gamma_d: float,
        lam: float,
        sigma: float,
        q_max,
        q_min,
        q_neutral,
        parent_dim: int
    ):
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.lam = lam
        self.sigma = sigma
        self.q_max = q_max
        self.q_min = q_min
        self.q_neutral = q_neutral
        
        super().__init__(name, parent_dim, calc_mappings, parent_dim)
    
    
    def pullback(self):
        if not self.isMulti:
            super().pullback()
        else:
            self.calc_rmp_func()
    
    def calc_rmp_func(self,):
        xi = np.empty((self.dim, 1))
        self.M.fill(0)

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