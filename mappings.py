
import numpy as np
from numpy import linalg as LA
from numba import njit

class Identity:
    """恒等写像"""
    def phi(self, x):
        return x.copy()
    
    def velocity(self, J, x_dot):
        return J @ x_dot
    
    def J(self, x):
        return np.eye(x.shape[0])
    
    def J_dot(self, x, x_dot):
        return np.zeros((x.shape[0], x.shape[0]))

    def calc_all(self, q, dq):
        x = self.phi(q)
        J = self.J(q)
        x_dot = self.velocity(J, dq)
        J_dot = self.J_dot(q, dq)
        return x, x_dot, J, J_dot

class Translation(Identity):
    """平行移動"""
    def __init__(self, g, g_dot):
        self.g = g
        self.g_dot = g_dot
        self.__J = np.eye(g.shape[0])
        self.__J_dot = np.zeros((g.shape[0], g.shape[0]))
    
    
    def phi(self, x):
        return x - self.g
    
    def velocity(self, J, x_dot):
        return x_dot - self.g_dot

    def J(self, x):
        return self.__J
    
    def J_dot(self, x, x_dot):
        return self.__J_dot

    def calc_all(self, q, dq):
        x = self.phi(q)
        J = self.J(q)
        x_dot = self.velocity(J, dq)
        J_dot = self.J_dot(q, dq)
        return x, x_dot, J, J_dot

class Distance(Identity):
    """距離写像"""
    def __init__(self, o, o_dot):
        self.o = o
        self.o_dot = o_dot
    
    def phi(self, x) -> float:
        return LA.norm(x - self.o)
    
    def velocity(self, J, x_dot) -> float:
        return (J @ (x_dot - self.o_dot))[0,0]
    
    def J(self, x):
        return -(x-self.o).T / LA.norm(x-self.o)
    
    def J_dot(self, x, x_dot):
        return calc_distance_J_dot(x, x_dot, self.o, self.o_dot)

    def calc_all(self, q, dq):
        x = self.phi(q)
        J = self.J(q)
        x_dot = self.velocity(J, dq)
        J_dot = self.J_dot(q, dq)
        return x, x_dot, J, J_dot

@njit('f8[:,:](f8[:,:], f8[:,:], f8[:,:], f8[:,:])', cache=True)
def calc_distance_J_dot(x, x_dot, o, o_dot):
    return -((x_dot-o_dot).T - (x-o).T*(1/LA.norm(x-o)*(x-o).T @ (x_dot-o_dot))) / LA.norm(x-o)**2



if __name__ == "__main__":
    pass