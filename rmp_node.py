"""基本のノード
"""
from __future__ import annotations  #一番初めに載せないとエラー?


import numpy as np
from numpy import linalg as LA
from typing import Union

from mappings import Identity


class Node:
    """node base and control point's node"""
    
    def __init__(
        self,
        name: str, dim: int, parent: Union[Node, None],
        mappings: Union[Identity, None],
        parent_dim: Union[int, None]=None,
    ):
        self.name = name
        self.dim= dim
        self.parent = parent
        self.mappings = mappings
        self.children: list[Node] = []
        self.isMulti = False
        

        self.x = np.empty((self.dim, 1))
        self.x_dot = np.empty((self.dim, 1))
        self.f = np.empty((self.dim, 1))
        self.M = np.empty((self.dim, self.dim))
        if parent is not None and parent_dim is None:
            self.J = np.empty((self.dim, parent.dim))
            self.J_dot = np.empty((self.dim, parent.dim))
        elif parent is None and parent_dim is not None:
            self.J = np.empty((self.dim, parent_dim))
            self.J_dot = np.empty((self.dim, parent_dim))
        else:
            assert False
    
    
    def add_child(self, child: Node):
        child.isMulti = self.isMulti
        self.children.append(child)
    
    
    def print_all_state(self,):
        self.print_state()
        for child in self.children:
            child.print_all_state()
    
    
    def pushforward(self):
        """push ノード"""
        for child in self.children:
            assert child.mappings is not None
            child.x = child.mappings.phi(self.x)
            child.J = child.mappings.J(self.x)
            child.x_dot = child.mappings.velocity(child.J, self.x_dot)
            child.J_dot = child.mappings.J_dot(self.x, self.x_dot)
            child.pushforward()
    
    
    def pullback(self):
        #print(self.name, ", pullback now")
        self.f.fill(0)
        self.M.fill(0)
        for child in self.children:
            child.pullback()
        
        if not self.isMulti:
            assert self.parent is not None
            self.parent.f += self.J.T @ (self.f - self.M @ self.J_dot @ self.parent.x_dot)
            self.parent.M += self.J.T @ self.M @ self.J
    
    
    def solve(self, parent_x, parent_x_dot):
        """並列処理用"""
        assert self.isMulti == True
        #time.sleep(1)
        
        # 自分をpush
        assert self.mappings is not None
        self.x = self.mappings.phi(parent_x)
        self.J = self.mappings.J(parent_x)
        self.x_dot = self.mappings.velocity(self.J, parent_x_dot)
        self.J_dot = self.mappings.J_dot(parent_x, parent_x_dot)

        self.pushforward()
        self.pullback()
        
        pulled_f = self.J.T @ (self.f - self.M @ self.J_dot @ parent_x_dot)
        pulled_M = self.J.T @ self.M @ self.J
        
        #print(self.name, "done!")
        return pulled_f, pulled_M
    

    def print_state(self,):
        print("name = ", self.name)
        print("parent =", self.parent.name if self.parent is not None else 'None')
        print("type = ", self.__class__.__name__)
        
        if self.children is None:
            print("children = ", self.children)
        else:
            print("children = ", end="")
            for child in self.children:
                print(child.name, end=", ")
            print("")
        print("x = \n", self.x)
        print("x_dot = \n", self.x_dot)
        if self.parent is not None:
            print("J = \n", self.J)
            print("J_dot = \n", self.J_dot)
        print("f = \n", self.f)
        print("M = \n", self.M)
        print("")


    def set_state(self, x, x_dot):
        self.x = x
        self.x_dot = x_dot


def multi_solve(child: Node, q, q_dot):
    """並列処理用"""
    print(child.name,"begin")
    #time.sleep(5)
    #print("  child = ", [c.name for c in child.children])
    
    f, M = child.solve(q, q_dot)
    print("    ", child.name, "done!")
    return f, M





class Root(Node):
    def __init__(self, dim: int, isMulti:bool=False):
        super().__init__(
            name = "root",
            parent = None,
            dim = dim,
            mappings = None,
        )

        self.x_ddot = np.empty((dim, 1))
        self.isMulti = isMulti
    
    
    def set_state(self, q, q_dot):
        self.x = q
        self.x_dot = q_dot
    
    
    def update_state(self, dt: float):
        self.x += self.x_dot * dt
        self.x_dot += self.x_ddot * dt
    
    
    def pullback(self):
        # 初期化
        self.f.fill(0)
        self.M.fill(0)
        
        # pullback開始
        for child in self.children:
            child.pullback()
    
        #print(self.name, "done")
    
    
    def resolve(self,) -> None:
        self.x_ddot = LA.pinv(self.M) @ self.f
    
    
    def solve(self, q=None, q_dot=None, dt=None, g=None, o_s=None):
        """resolvwの反対"""
        self.set_state(q, q_dot)
        if self.isMulti == False:
            self.pushforward()
            self.pullback()
            self.resolve()
            return self.x_ddot
        else:
            pass





if __name__ == "__main__":
    pass