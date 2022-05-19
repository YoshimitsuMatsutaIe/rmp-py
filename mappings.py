import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
import numpy.typing as npt


class Identity:
    """恒等写像"""
    def phi(self, x: npt.NDArray[np.float64]):
        return x.copy()
    
    def velocity(self, J: NDArray[np.float64], x_dot: NDArray[np.float64]) -> NDArray[np.float64]:
        return J @ x_dot
    
    def J(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.eye(x.shape[0])
    
    def J_dot(self, x: NDArray[np.float64], x_dot: NDArray[np.float64]):
        return np.zeros((x.shape[0], x.shape[0]))


class Translation(Identity):
    def __init__(self, g: NDArray[np.float64], g_dot: NDArray[np.float64]):
        self.g = g
        self.g_dot = g_dot
    
    def phi(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x - self.g
    
    def velocity(self, J: NDArray[np.float64], x_dot: NDArray[np.float64]) -> NDArray[np.float64]:
        return x_dot - self.g_dot


class Distance(Identity):
    """距離写像"""
    def __init__(self, o: NDArray[np.float64], o_dot: NDArray[np.float64]):
        self.o = o
        self.o_dot = o_dot
    
    def phi(self, x: NDArray[np.float64]) -> float:
        #print("s = ", LA.norm(x - self.o),)# end="  ")
        return LA.norm(x - self.o)
    
    def velocity(self, J: NDArray[np.float64], x_dot: NDArray[np.float64]):
        #print("ds = ", J @ (x_dot - self.o_dot), end="  ")
        #print("J = ", J)
        #print("x_dot = ", x_dot.T)
        return J @ (x_dot - self.o_dot)
    
    def J(self, x):
        return -(x-self.o).T / LA.norm(x-self.o)
    
    def J_dot(self, x: NDArray[np.float64], x_dot: NDArray[np.float64]):
        return -((x_dot-self.o_dot).T - (x-self.o).T*(1/LA.norm(x-self.o)*(x-self.o).T @ (x_dot-self.o_dot))) / LA.norm(x-self.o)**2



if __name__ == "__main__":
    pass