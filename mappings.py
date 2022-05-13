import numpy as np
from numpy import linalg as LA
import typing
import nptyping as npt


class Identity:
    """恒等写像"""
    def phi(self, x: npt.NDArray):
        return x.copy()
    
    def velocity(self, J: npt.NDArray, x_dot: npt.NDArray) -> npt.NDArray:
        return J @ x_dot
    
    def J(self, x: npt.NDArray) -> npt.NDArray:
        return np.eye(x.shape[0])
    
    def J_dot(self, x: npt.NDArray, x_dot: npt.NDArray):
        return np.zeros((x.shape[0], x.shape[0]))


class Translation(Identity):
    def __init__(self, g, g_dot):
        self.g: npt.NDArray = g
        self.g_dot: npt.NDArray = g_dot
    
    def phi(self, x: npt.NDArray):
        return x - self.g
    
    def velocity(self, J: npt.NDArray, x_dot: npt.NDArray):
        return x_dot - self.g_dot


class Distance(Identity):
    """距離写像"""
    
    def __init__(self, o: npt.NDArray, o_dot: npt.NDArray):
        self.o = o
        self.o_dot = o_dot
    
    def phi(self, x) -> float:
        return LA.norm(x-self.o)
    
    def velocity(self, J, x_dot):
        return J @ (x_dot - self.o_dot)
    
    def J(self, x):
        return -(x-self.o).T / LA.norm(x-self.o)
    
    def J_dot(self, x, x_dot):
        return -((x_dot-self.o_dot).T - (x-self.o).T*(1/LA.norm(x-self.o)*(x-self.o).T @ (x_dot-self.o_dot))) / LA.norm(x-self.o)**2



if __name__ == "__main__":
    pass