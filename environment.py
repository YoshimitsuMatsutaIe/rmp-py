import numpy as np
np.random.seed(0)  # 固定

from math import cos, sin, tan, pi
import matplotlib.pyplot as plt


def rotate3d(alpha: float, beta: float, gamma: float):
    """3次元回転行列"""
    rx = np.array([
        [1, 0, 0],
        [0, cos(alpha), -sin(alpha)],
        [0, sin(alpha), cos(alpha)],
    ])
    ry = np.array([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)],
    ])
    rz = np.array([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1],
    ])
    return rx @ ry @ rz


def set_point(x, y, z=None):
    """点を置く"""
    if z is None:
        return [np.array([[x, y]]).T]
    else:
        return [np.array([[x, y, z]]).T]


def set_sphere(n: int, r: float, x: float, y: float, z=None):
    """
    
    r : 半径
    center : 中心
    n : 点の数
    """
    
    if z is None:
        center = np.array([[x, y]]).T
    else:
        center = np.array([[x, y, z]]).T
    
    obs = []
    rand = np.random.RandomState(123)
    for i in range(n):
        alpha = np.arccos(rand.uniform(-1, 1))
        beta = rand.uniform(0, 2*pi)
        
        if z is None:
            x = r * cos(beta)
            y = r * sin(beta)
            obs.append(np.array([[x, y]]).T + center)
        else:
            x = r * sin(alpha) * cos(beta)
            y = r * sin(alpha) * sin(beta)
            z = r * cos(alpha)
            obs.append(np.array([[x, y, z]]).T + center)
    
    return obs


def set_cylinder(n: int, r, L, x, y, z, alpha=0.0, beta=0.0, gamma=0.0,):
    """円筒を設置
    
    r : 半径
    L : 長さ
    n : 点の数
    alpha : 回転 [degree]
    beta : 回転 [degree]
    gamma : 回転 [degree]
    """
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    obs = []
    rand = np.random.RandomState(123)
    for i in range(n):
        _alpha = rand.uniform(0, 2*pi)
        X = np.array([
            [r * cos(_alpha)],
            [r * sin(_alpha)],
            [rand.uniform(-L/2, L/2)],
            ])
        obs.append(R @ X + np.array([[x, y, z]]).T)
    
    return obs


def set_field(n: int, lx, ly, x, y, z, alpha=0, beta=0, gamma=0):
    """面を表現"""
    
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    center = np.array([[x, y, z]]).T
    obs = []
    
    rand = np.random.RandomState(123)
    for i in range(n):
        X = np.array([
            [rand.uniform(-1, 1) * lx/2],
            [rand.uniform(-1, 1) * ly/2],
            [0],
            ])
        obs.append(R @ X + center)
    
    return obs



def set_box(n: int, lx, ly, lz, x, y, z, alpha=0, beta=0, gamma=0):
    """箱型障害物"""
    
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    center = np.array([[x, y, z]]).T
    
    s1 = lx * ly
    s2 = lx * lz
    s3 = ly * lz
    
    sum_s = s1 + s2 + s3
    
    n1 = int(n * s1 / sum_s / 2)
    n2 = int(n * s2 / sum_s / 2)
    n3 = int(n * s3 / sum_s / 2)
    
    obs = []
    obs += set_field(n1, lx=lx, ly=ly, x=0, y=0, z=lz/2)
    obs += set_field(n1, lx=lx, ly=ly, x=0, y=0, z=-lz/2)
    obs += set_field(n2, lx=lx, ly=lz, x=0, y=ly/2, z=0, alpha=90)
    obs += set_field(n2, lx=lx, ly=lz, x=0, y=-ly/2, z=0, alpha=90)
    obs += set_field(n3, lx=lz, ly=ly, x=lx/2, y=0, z=0, beta=90)
    obs += set_field(n3, lx=lz, ly=ly, x=-lx/2, y=0, z=0, beta=90)
    
    for i, o in enumerate(obs):
        obs[i] = R @ o + center
    
    return obs



def set_cubbie(n: int, lx, ly, lz, x, y, z, alpha=0, beta=0, gamma=0):
    """キャビネット"""
    
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    center = np.array([[x, y, z]]).T
    
    s1 = lx * ly
    s2 = lx * lz
    s3 = ly * lz
    
    sum_s = s1 + s2 + s3
    
    n1 = int(n * s1 / sum_s / 2)
    n2 = int(n * s2 / sum_s / 2)
    n3 = int(n * s3 / sum_s / 2)
    
    obs = []
    obs += set_field(n1, lx=lx, ly=ly, x=0, y=0, z=lz/2)
    obs += set_field(n1, lx=lx, ly=ly, x=0, y=0, z=-lz/2)
    obs += set_field(n2, lx=lx, ly=lz, x=0, y=ly/2, z=0, alpha=90)
    #obs += set_field(n2, lx=lx, ly=lz, x=0, y=-ly/2, z=0, alpha=90)
    obs += set_field(n3, lx=lz, ly=ly, x=lx/2, y=0, z=0, beta=90)
    obs += set_field(n3, lx=lz, ly=ly, x=-lx/2, y=0, z=0, beta=90)
    
    for i, o in enumerate(obs):
        obs[i] = R @ o + center
    
    return obs




if __name__ == '__main__':
    pass