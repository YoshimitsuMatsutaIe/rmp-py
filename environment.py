
import numpy as np
np.random.seed(0)  # 固定

from math import cos, sin, tan, pi



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




def set_sphere_rand(n: int, r, x, y, z=None):
    obs = []
    rand = np.random.RandomState(123)
    if z is None:
        for _ in range(n):
            beta = rand.uniform(0, 2*pi)
            X = r * cos(beta) + x
            Y = r * sin(beta) + y
            obs.append(np.array([[X], [Y]]))
    else:
        for _ in range(n):
            alpha = np.arccos(rand.uniform(-1, 1))
            beta = rand.uniform(0, 2*pi)
            X = r * sin(alpha) * cos(beta) + x
            Y = r * sin(alpha) * sin(beta) + y
            Z = r * cos(alpha) + z
            obs.append(np.array([[X], [Y], [Z]]))
    
    return obs


def set_sphere(d: float, r, x, y, z=None):
    obs = []
    if z is None:
        beta = np.arange(0, 2*pi, d/r)
        for b in beta:
            obs.append(np.array([[r*cos(b) + x], [r*sin(b) + y]]))
    else:
        N = int(4 * r**2/d**2)
        theta = pi
        phi = 0
        for i in range(1, N):
            if i == 1:
                pass
            elif i == N:
                theta = 0
                phi = pi
            else:
                h = 2*(i-1)/(N-1) -1
                theta = np.arccos(h)
                phi += 3.6/np.sqrt(N)*1/(np.sqrt(1-h**2))
            X = r * sin(theta) * cos(phi) + x
            Y = r * sin(theta) * sin(phi) + y
            Z = r * cos(theta) + z
            obs.append(np.array([[X], [Y], [Z]]))

    return obs




# def set_2d_bowl(n: int, r, x, y, alpha=0.0):
    


def set_cylinder_rand(n: int, r, L, x, y, z, alpha=0.0, beta=0.0, gamma=0.0,):
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


def set_cylinder(d: float, r, L, x, y, z, alpha=0.0, beta=0.0, gamma=0.0):
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    obs = []
    
    theta = np.arange(0, 2*pi, d/r)
    z_axis = np.arange(-L/2, L/2, d)
    theta, z_axis = np.meshgrid(theta, z_axis)
    theta = np.ravel(theta)
    z_axis = np.ravel(z_axis)

    for i in range(theta.size):
        X = np.array([
            [r * cos(theta[i])],
            [r * sin(theta[i])],
            [z_axis[i]],
            ])
        obs.append(R @ X + np.array([[x, y, z]]).T)
    
    return obs


def set_field_rand(n: int, lx, ly, x, y, z, alpha=0, beta=0, gamma=0):
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


def set_field(d: float, lx, ly, x, y, z, alpha=0, beta=0, gamma=0):
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    center = np.array([[x, y, z]]).T
    xx, yy = np.meshgrid(np.arange(-lx/2, lx/2, d), np.arange(-ly/2, ly/2, d))
    obs_ = np.vstack(
        [np.ravel(xx), np.ravel(yy), np.zeros(xx.size)]
    )
    obs = []
    for i in range(xx.size):
        obs.append(R @ obs_[:, i:i+1] + center)
    
    return obs


def set_box_rand(n: int, lx, ly, lz, x, y, z, alpha=0, beta=0, gamma=0):
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
    obs += set_field_rand(n1, lx=lx, ly=ly, x=0, y=0, z=lz/2)
    obs += set_field_rand(n1, lx=lx, ly=ly, x=0, y=0, z=-lz/2)
    obs += set_field_rand(n2, lx=lx, ly=lz, x=0, y=ly/2, z=0, alpha=90)
    obs += set_field_rand(n2, lx=lx, ly=lz, x=0, y=-ly/2, z=0, alpha=90)
    obs += set_field_rand(n3, lx=lz, ly=ly, x=lx/2, y=0, z=0, beta=90)
    obs += set_field_rand(n3, lx=lz, ly=ly, x=-lx/2, y=0, z=0, beta=90)
    
    for i, o in enumerate(obs):
        obs[i] = R @ o + center
    
    return obs


def set_box(d: float, lx, ly, lz, x, y, z, alpha=0, beta=0, gamma=0):
    """箱型障害物"""
    
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    center = np.array([[x, y, z]]).T

    obs = []
    obs += set_field(d, lx=lx, ly=ly, x=0, y=0, z=lz/2)
    obs += set_field(d, lx=lx, ly=ly, x=0, y=0, z=-lz/2)
    obs += set_field(d, lx=lx, ly=lz, x=0, y=ly/2, z=0, alpha=90)
    obs += set_field(d, lx=lx, ly=lz, x=0, y=-ly/2, z=0, alpha=90)
    obs += set_field(d, lx=lz, ly=ly, x=lx/2, y=0, z=0, beta=90)
    obs += set_field(d, lx=lz, ly=ly, x=-lx/2, y=0, z=0, beta=90)
    
    for i, o in enumerate(obs):
        obs[i] = R @ o + center
    
    return obs



def set_cubbie_rand(n: int, lx, ly, lz, x, y, z, alpha=0, beta=0, gamma=0):
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
    obs += set_field_rand(n1, lx=lx, ly=ly, x=0, y=0, z=lz/2)
    obs += set_field_rand(n1, lx=lx, ly=ly, x=0, y=0, z=-lz/2)
    obs += set_field_rand(n2, lx=lx, ly=lz, x=0, y=ly/2, z=0, alpha=90)
    obs += set_field_rand(n3, lx=lz, ly=ly, x=lx/2, y=0, z=0, beta=90)
    obs += set_field_rand(n3, lx=lz, ly=ly, x=-lx/2, y=0, z=0, beta=90)
    
    for i, o in enumerate(obs):
        obs[i] = R @ o + center
    
    return obs


def set_cubbie(d: float, lx, ly, lz, x, y, z, alpha=0, beta=0, gamma=0):
    """キャビネット"""
    
    R = rotate3d(np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma))
    center = np.array([[x, y, z]]).T
    

    obs = []
    obs += set_field(d, lx=lx, ly=ly, x=0, y=0, z=lz/2)
    obs += set_field(d, lx=lx, ly=ly, x=0, y=0, z=-lz/2)
    obs += set_field(d, lx=lx, ly=lz, x=0, y=ly/2, z=0, alpha=90)
    obs += set_field(d, lx=lz, ly=ly, x=lx/2, y=0, z=0, beta=90)
    obs += set_field(d, lx=lz, ly=ly, x=-lx/2, y=0, z=0, beta=90)
    
    for i, o in enumerate(obs):
        obs[i] = R @ o + center
    
    return obs


def set_obstacle(obs_params: list[dict]):
    obstacle = []
    for obs_param in obs_params:
        type_ = obs_param["type"]
        param_ = obs_param["param"]
        
        if type_ == "cylinder":
            obstacle.extend(set_cylinder(**param_))
        elif type_ == "sphere":
            obstacle.extend(set_sphere(**param_))
        elif type_ == "field":
            obstacle.extend(set_field(**param_))
        elif type_ == "box":
            obstacle.extend(set_box(**param_))
        elif type_ == "cubbie":
            obstacle.extend(set_cubbie(**param_))
        elif type_ == "point":
            obstacle.extend(set_point(**param_))
        elif type_ == "cylinder_rand":
            obstacle.extend(set_cylinder_rand(**param_))
        elif type_ == "sphere_rand":
            obstacle.extend(set_sphere_rand(**param_))
        elif type_ == "field_rand":
            obstacle.extend(set_field_rand(**param_))
        elif type_ == "box_rand":
            obstacle.extend(set_box_rand(**param_))
        elif type_ == "cubbie_rand":
            obstacle.extend(set_cubbie_rand(**param_))
        else:
            assert False
    return obstacle



if __name__ == '__main__':
    
    # o = set_field(0.1, 1, 2, 1, 2, 3, 45, 30, 60)

    # import matplotlib.pyplot as plt

    # o = np.concatenate(o, axis=1)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(o[0,:], o[1,:], o[2,:])

    # fig.savefig("hoge.png")


    o = set_cylinder(0.05, 0.2, 1, 2, 1, 1, 60, 30, 45)
    #o = set_sphere_rand(100, 1, 1, 2, 1)

    import matplotlib.pyplot as plt

    o = np.concatenate(o, axis=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(o[0,:], o[1,:], o[2,:])
    

    fig.savefig("hoge.png")

