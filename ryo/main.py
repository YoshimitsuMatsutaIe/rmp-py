import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time
from math import sin, cos, sqrt, pi
from numba import njit

from itertools import product

@njit("f8[:,:](f8, f8, f8, f8)", cache=True)
def HTM(theta, a, d, alpha):
    "同時変換行列"
    return np.array([
        [cos(theta), -sin(theta), 0.0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -d*sin(alpha)],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), d*cos(alpha)],
        [0.0, 0.0, 0.0, 1.0],
    ])

# def trans(x, y, z):
#     return np.array([
#         [1, 0, 0, x],
#         [0, 1, 0, y],
#         [0, 0, 1, z],
#         [0, 0, 0, 1]
#     ])

# def rotx(angle):
#     return np.array([
#         [1, 0, 0, 0],
#         [0, cos(angle), -sin(angle), 0],
#         [0, sin(angle), cos(angle), 0],
#         [0, 0, 0, 1]
#     ])

# def rotz(angle):
#     return np.array([
#         [cos(angle), -sin(angle), 0, 0],
#         [sin(angle), cos(angle), 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])

@njit("List(f8[:,:])(f8[:])", cache=True)
def f_kinematics(theta):
    a = [0, 0, 264, 236, 0, 0]
    alpha = [0, pi/2, 0, 0, pi/2, -pi/2]
    d = [144, 0, 0, 106, 114, 67]
    
    T_local = [HTM(theta[i], a[i], d[i], alpha[i]) for i in range(6)]
    # T_local = [
    #     trans(a[i],0,0) @ rotx(alpha[i]) @ trans(0,0,d[i]) @ rotz(theta[i]) for i in range(6)
    # ]
    T_global = []
    for i, T in enumerate(T_local):
        if i == 0:
            T_global.append(T)
        else:
            T_global.append(T_global[-1] @ T)
    
    return T_global


def F_attractive_enerfy(goal, theta):
    Kat = 1
    
    _, _, _, _, _, T6 = f_kinematics(theta)
    x, y, z = T6[0:3, 3]
    
    goal_x, goal_y, goal_z = goal
    
    d = sqrt((x - goal_x)**2 + (y - goal_y)**2 + (z - goal_z)**2)
    Fat = 0.5 * Kat * d**2
    return Fat


def F_repulsive_energy(obstacle, R, goal, theta):
    Kre = 1
    d0 = R + 100
    fre = 0
    goal_x, goal_y, goal_z = goal
    obs_x, obs_y, obs_z = obstacle
    for T in f_kinematics(theta):
        x, y, z = T[0:3, 3]
        rat = sqrt((x - goal_x)**2 + (y - goal_y)**2 + (z - goal_z)**2)
        rre = sqrt((x - obs_x)**2 + (y - obs_y)**2 + (z - obs_z)**2)
        if rre > 300:
            fre += 0
        else:
            fre += 0.5 * Kre * (1/rre - 1/d0)**2 * rat**2
    
    return fre


def CollisionTest(theta, R, C):
    Ts = f_kinematics(theta)
    for T in Ts:
        d = LA.norm(T[0:3, 3] - C)
        if d < R+100:
            return True
        else:
            pass
    return False


def main():
    print("running...")
    goal = [-400, 200, -300]
    start_angle = np.array([pi/2, pi/2, -pi/2, 0, 0, 0])
    step_angle = 2
    
    theta = start_angle
    
    obs_R = 100
    obs_C = np.array([-100, 336, -50])
    
    F_list = []
    theta_list = []
    path_list = [start_angle]
    Fatt_list = []
    Fatt_min_list = []
    Frel_list = []
    Frel_min_list = []
    
    t0 = time.perf_counter()
    while True:
        qs = np.vstack([
            theta-step_angle*pi/180,
            theta,
            theta+step_angle*pi/180
        ])
        
        for theta1, theta2, theta3, theta4, theta5 in product(*qs.T[:5].tolist()):
            tmp_theta = np.array([theta1, theta2, theta3, theta4, theta5, theta[5]])
            if CollisionTest(tmp_theta, obs_R, obs_C):
                k = 1
            else:
                Fatt = F_attractive_enerfy(goal, tmp_theta)
                Frel = F_repulsive_energy(obs_C, obs_R, goal, tmp_theta)
                Frel_list.append(Frel)
                Fatt_list.append(Fatt)
                F_list.append(Fatt + Frel)
                theta_list.append(tmp_theta)
    
        F_min_id = F_list.index(min(F_list))
        path_list.append(theta_list[F_min_id])
        Fatt_min_list.append(Fatt_list[F_min_id])
        Frel_min_list.append(Frel_list[F_min_id])
        theta = theta_list[F_min_id]
        
        if LA.norm(path_list[-1] - path_list[-2]) < 1e-3:
            break
    
    print("done!\nruntime = {0} [s]".format(time.perf_counter() - t0))
    
    
    ### animation ###
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection="3d")
    
    #obs ball 1
    u = np.linspace(0, 2*pi, 10)
    v = np.linspace(0, pi, 10)
    x1 = obs_R * np.outer(np.cos(u), np.sin(v)) + obs_C[0]
    y1 = obs_R * np.outer(np.sin(u), np.sin(v)) + obs_C[1]
    z1 = obs_R * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_C[2]
    
    ee_path = []
    for i, theta in enumerate(path_list):
        _, _, _, _, _, T6 = f_kinematics(theta)
        x, y, z = T6[0:3, 3]
        if i == 0:
            ee_path = [[x], [y], [z]]
        else:
            ee_path[0].append(x)
            ee_path[1].append(y)
            ee_path[2].append(z)
    
    
    def update(i):
        ax.cla()
        
        Ts = f_kinematics(path_list[i])
        joint = []
        for j in range(3):
            tmp = []
            for T in Ts:
                tmp.append(T[j, 3])
            joint.append(tmp)
        
        ax.plot(joint[0], joint[1], joint[2], "o-", label="arm")
        ax.plot(ee_path[0][:i], ee_path[1][:i], ee_path[2][:i], label="end effector ")
        ax.scatter(goal[0], goal[1], goal[2], marker="*", label="goal", color='#ff7f00', s=100)
        ax.scatter(obs_C[0], obs_C[1], obs_C[2], label="obs")
        ax.plot_surface(x1, y1, z1,color="C7",alpha=0.3,rcount=100, ccount=100, antialiased=False,)
        
        ax.grid(True)
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(-500, 500)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_box_aspect((1,1,1))
        ax.legend()
        
        return
    
    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(path_list), 5)
    )
    
    plt.show()
    return





if __name__ == "__main__":
    main()