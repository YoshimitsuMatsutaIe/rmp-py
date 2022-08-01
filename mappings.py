import numpy as np
from numpy import linalg as LA



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


class Translation(Identity):
    def __init__(self, g, g_dot):
        self.g = g
        self.g_dot = g_dot
    
    def phi(self, x):
        return x - self.g
    
    def velocity(self, J, x_dot):
        return x_dot - self.g_dot


class Distance(Identity):
    """距離写像"""
    def __init__(self, o, o_dot):
        self.o = o
        self.o_dot = o_dot
    
    def phi(self, x) -> float:
        s = LA.norm(x - self.o)
        if s < 0.01:
            print("s = ", s)
        return s
    
    def velocity(self, J, x_dot):
        #print("ds = ", J @ (x_dot - self.o_dot), end="  ")
        #print("J = ", J)
        #print("x_dot = ", x_dot.T)
        return J @ (x_dot - self.o_dot)
    
    def J(self, x):
        return -(x-self.o).T / LA.norm(x-self.o)
    
    def J_dot(self, x, x_dot):
        return -((x_dot-self.o_dot).T - (x-self.o).T*(1/LA.norm(x-self.o)*(x-self.o).T @ (x_dot-self.o_dot))) / LA.norm(x-self.o)**2



if __name__ == "__main__":
    pass