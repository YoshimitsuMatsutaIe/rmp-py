"""ryo's pathlanning"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time
from numba import njit
from tqdm import tqdm
from math import pi

from itertools import product

from robot_utils import get_robot_model, KinematicsAll



def F_attractive_enerfy(Kat, goal, x):
    d = LA.norm(x - goal)
    Fat = 0.5 * Kat * d**2
    return Fat


def F_repulsive_energy(Kre: float, R, obs_Cs, goal, x_all, ee_x):
    d0 = R + 0.1
    rat = LA.norm(ee_x - goal)
    
    fre = 0
    for x in x_all:
        rre_s = LA.norm(x - obs_Cs, axis=0)
        rre_s = rre_s[rre_s <= R]
        fre += np.sum((1/rre_s - 1/d0)**2)
    
    return fre * 0.5 * Kre * rat**2


def planning(
    robot_name: str,
    q_init, q_step,
    goal, obs_R, obs_Cs,
    Kat=1.0, Kre=1.0,
    q_step_n=1, max_step=10000
):
    print("running...")

    km = KinematicsAll(robot_name)

    F_list = []
    q_list = []
    q_path_list = [q_init]
    # Fatt_list = []
    # Fatt_min_list = []
    # Frel_list = []
    # Frel_min_list = []
    
    obs_Cs_ = np.concatenate(obs_Cs, axis=1)
    
    q = q_init
    
    t0 = time.perf_counter()
    
    for i in tqdm(range(max_step)):
        q_ = [q]
        for j in range(q_step_n):
            q_.append(q - q_step*pi/180)
            q_.append(q + q_step*pi/180)
        
        q_set = np.concatenate(q_, axis=1)
        
        for tmp_q in product(*(q_set).tolist()):
            #print("tmp_q = ", tmp_q)
            q = np.array([tmp_q]).T
            
            x_all = km.calc_cpoint_position_all(q)
            x_all_c = np.concatenate(x_all, axis=1)
            
            for o in obs_Cs:
                if np.any(LA.norm(x_all_c - o, axis=0) < obs_R):
                    break
                else:
                    k = 1
                    continue
            
            ee_x = km.calc_ee_position(q)
            Fatt = F_attractive_enerfy(Kat, goal, ee_x)
            Frel = F_repulsive_energy(Kre, obs_R, obs_Cs_, goal, x_all, ee_x)
            F = Fatt + Frel
            #print(F)
            # Frel_list.append(Frel)
            # Fatt_list.append(Fatt)
            F_list.append(F)
            q_list.append(q)
    
        F_min_id = F_list.index(min(F_list))
        q_path_list.append(q_list[F_min_id])
        # Fatt_min_list.append(Fatt_list[F_min_id])
        # Frel_min_list.append(Frel_list[F_min_id])
        q = q_list[F_min_id]
        
        if LA.norm(q_path_list[-1] - q_path_list[-2]) < 1e-3:
            break
    
    print("done!\nruntime = {0} [s]".format(time.perf_counter() - t0))
    
    return q_path_list



if __name__ == "__main__":
    
    robot_name = "sice"
    K = KinematicsAll(robot_name)
    q_init = np.zeros((4, 1))
    q_step = 1
    goal = np.array([[0.5, 2.5]]).T
    obs_R = 1
    obs_Cs = [
        np.array([[3.0, 2.0]]).T,
        np.array([[3.0, 2.1]]).T,
        np.array([[3.0, 2.3]]).T,
        # np.array([[3.0, 2.4]]).T,
        # np.array([[3.0, 2.5]]).T,
        # np.array([[3.0, 2.6]]).T
    ]
    
    q_path_list = planning(robot_name, q_init, q_step, goal, obs_R, obs_Cs)

    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()#projection="3d")
    
    # #obs ball 1
    # u = np.linspace(0, 2*pi, 15)
    # v = np.linspace(0, pi, 15)
    # x1 = obs_R * np.outer(np.cos(u), np.sin(v)) + obs_C[0]
    # y1 = obs_R * np.outer(np.sin(u), np.sin(v)) + obs_C[1]
    # z1 = obs_R * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_C[2]
    
    # ee_path = []
    # for i, theta in enumerate(q_path_list):
    #      = f_kinematics(theta)
    #     x, y, z = T6[0:3, 3]
    #     if i == 0:
    #         ee_path = [[x], [y], [z]]
    #     else:
    #         ee_path[0].append(x)
    #         ee_path[1].append(y)
    #         ee_path[2].append(z)
    
    obs = np.concatenate(obs_Cs, axis=1)
    
    def update(i):
        ax.cla()
        
        xs = K.calc_cpoint_position_all(q_path_list[i])
        xs = np.concatenate(xs, axis=1)
        
        ax.scatter(xs[0, :], xs[1, :], label="arm")
        # ax.plot(ee_path[0][:i], ee_path[1][:i], ee_path[2][:i], label="end effector ")
        ax.scatter(goal[0,0], goal[1,0], marker="*", label="goal", color='#ff7f00', s=100)
        ax.scatter(obs[0, :], obs[1, :], marker="+", label="obs")
        # ax.plot_surface(x1, y1, z1,color="C7",alpha=0.3,rcount=100, ccount=100, antialiased=False,)
        ax.set_title(str(i))
        ax.grid(True)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        #ax.set_zlim(-500, 500)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        #ax.set_zlabel('Z [m]')
        #ax.set_box_aspect((1,1,1))
        ax.set_aspect('equal')
        ax.legend()
        
        return
    
    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(q_path_list))
    )
    
    plt.show()