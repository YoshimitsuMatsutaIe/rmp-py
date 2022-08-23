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
        else:
            self.xi_func = attractor_xi_3d.f
        
        super().__init__(name, dim, parent, calc_mappings,)
    
    
    def calc_rmp_func(self,):
        print("error = ", self.x.T)
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
        #print(".", end='')
        self.M = self.__inertia_matrix()
        self.f = self.__force()
        if abs(self.f[0, 0]) > 0.1:
            print("obs_f = ", self.f[0,0], ", obs_m = ", self.M[0, 0])
    
    def __w2(self, s):
        if self.rw - s > 0:
            return (self.rw - s)**2 / s
        else:
            return 0
    
    def __w2_dot(self, s):
        if self.rw - s > 0:
            return (-2*(self.rw-s)*s + (self.rw-s)) / s**2
        else:
            return 0
    
    def __u2(self, s_dot):
        if s_dot < 0:
            return 1 - exp(-s_dot**2 / (2*self.sigma**2))
        else:
            return 0
    
    def __u2_dot(self, s_dot):
        #print(s_dot)
        if s_dot < 0:
            return -exp(s_dot**2 / (2*self.sigma**2)) * (-s_dot/self.sigma**3)
        else:
            return 0
    
    def __delta(self, s_dot):
        return self.__u2(s_dot) + 1/2 * s_dot * self.__u2_dot(s_dot)
    
    def __xi(self, s, s_dot):
        return 1/2 * self.__u2(s_dot) * self.__w2_dot(s) * s_dot**2
    
    def __grad_phi(self, s):
        return self.gain * self.__w2(s) * self.__w2_dot(s)
    
    def __inertia_matrix(self):
        return np.array([[
            self.__w2(self.x) * self.__delta(self.x_dot[0,0])
        ]])
    
    def __force(self):
        return np.array([[
            -self.__grad_phi(self.x) - self.__xi(self.x, self.x_dot[0,0])
        ]])



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
        self.M = self.__inertia_matrix()
        self.f = self.__force()
        print("jl_f = ", self.f.T)
    
    def __alpha_upper(self, q_dot):
        return 1 - exp(-max(q_dot, 0)**2 / (2*self.sigma**2))
    
    def __alpha_lower(self, q_dot):
        return 1 - exp(-min(q_dot, 0)**2 / (2*self.sigma**2))
    
    def __s(self, q, qu, ql):
        return (q - ql) / (qu - ql)
    
    def __s_dot(self, qu, ql):
        return 1 / (qu - ql)
    
    def __d(self, s):
        return 4*s*(1-s)
    
    def __d_dot(self, s, s_dot):
        return (4-8*s) * s_dot
    
    def __b(self, q, q_dot, qu, ql):
        s = self.__s(q, qu, ql)
        d = self.__d(s)
        au = self.__alpha_upper(q_dot)
        al = self.__alpha_lower(q_dot)
        #print(s, d, au, al)
        return s*(au*d + (1-au)) + (1-s)*(al*d + (1-al))
    
    def __b_dot(self, q, q_dot, qu, ql):
        s = self.__s(q, qu, ql)
        d = self.__d(s)
        au = self.__alpha_upper(q_dot)
        al = self.__alpha_lower(q_dot)
        s_dot = self.__s_dot(qu, ql)
        d_dot = self.__d_dot(s, s_dot)
        return (s_dot*(au*d + (1-au)) + s*d_dot) + -s_dot*(al * d + (1-al)) + (1-s) * d_dot
    
    def __a(self, q, q_dot, qu, ql):
        return self.__b(q, q_dot, qu, ql)**(-2)
    
    def __a_dot(self, q, q_dot, qu, ql):
        return -2*self.__b(q, q_dot, qu, ql)**(-3) * self.__b_dot(q, q_dot, qu ,ql)
    
    def __xi(self,):
        xi = []
        for i in range(self.dim):
            _xi =  1/2 * self.__a_dot(
                q=self.x[i,0],
                q_dot=self.x_dot[i,0],
                qu=self.q_max[i,0],
                ql=self.q_min[i,0]
            ) * self.x_dot[i,0]**2
            xi.append(_xi)
        return np.array([xi]).T
    
    def __inertia_matrix(self,):
        diags = []
        for i in range(self.dim):
            _s = self.lam * self.__a(
                self.x[i,0],
                self.x_dot[i,0],
                self.q_max[i,0],
                self.q_min[i,0])
            diags.append(_s)
        #print(diags)
        return np.diag(diags)
    
    def __force(self,):
        return self.M @ (self.gamma_p*(self.q_neutral - self.x) - self.gamma_d*self.x_dot) - self.__xi()



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