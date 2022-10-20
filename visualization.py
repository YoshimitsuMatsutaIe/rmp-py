"""ロボットアームをアニメ化する"""

import matplotlib.animation as anm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
import time
from functools import lru_cache  # これつけるとplt.show()でアニメーションがループしなくなる
import pickle
from typing import Union
from multiprocessing import Pool, cpu_count



def calc_scale(max_x, min_x, max_y, min_y,max_z=None, min_z=None):
    """軸範囲を計算"""
    
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    if max_z is None and min_z is None:
        max_range = max(max_x-min_x, max_y-min_y) * 0.5
        return (
            mid_x - max_range, mid_x + max_range,
            mid_y - max_range, mid_y + max_range
        )
    else:
        assert max_z is not None and min_z is not None
        max_range = max(max_x-min_x, max_y-min_y, max_z-min_z) * 0.5
        mid_z = (max_z + min_z) * 0.5
        return (
            mid_x - max_range, mid_x + max_range,
            mid_y - max_range, mid_y + max_range,
            mid_z - max_range, mid_z + max_range
        )



def make_data(
    q_s: list[list[float]],
    step,
    epoch_max,
    joint_phi_s,
    ee_phi=None,
    cpoint_phi_s=None,

):
    """描写したい点列を生成
    """
    
    #print(joint_phi_s)
    
    T_SIZE = epoch_max  # ループ回数
    q_real = np.array(q_s)
    q = np.array([q_[::step] for q_ in q_s])  # 縦にくっつくける
    
    joint_data = []
    for i in range(T_SIZE):
        temp = joint_phi_s(q[:, i:i+1])
        
        temp = np.concatenate(temp, axis=1)
        #print(temp)
        joint_data.append(temp.tolist())

    if ee_phi is not None:
        ee_data = []
        for i in range(len(q_s[0])):
            ee_data.append(ee_phi(q_real[:, i:i+1]))
            #print(ee_phi(q_real[:, i:i+1]).T)
        ee_data = np.concatenate(ee_data, axis=1)
        #print(ee_data)
    else:
        ee_data = None

    if cpoint_phi_s is not None:
        cpoint_data = []
        for i in range(T_SIZE):
            temp = cpoint_phi_s(q[:, i:i+1])
            temp_ = np.concatenate(temp, axis=1)
            cpoint_data.append(temp_.tolist())
    else:
        cpoint_data = None
    
    return q.T, joint_data, ee_data, cpoint_data




def make_animation(
    t_data: Union[list[float], list[int]],
    joint_data: list[list[list[float]]],
    epoch_max, step,
    q_data=None,
    ee_data=None,
    cpoint_data=None,
    goal_data=None,
    obs_data=None,
    is3D: bool=True,
    save_path: Union[str, None]=None,
):
    if is3D:
        return make_3d_animation(
            t_data,
            joint_data,
            epoch_max,
            step,
            q_data,
            ee_data,
            cpoint_data,
            goal_data,
            obs_data,
            save_path,
        )
    else:
        return make_2d_animation(
            t_data,
            joint_data,
            epoch_max, step,
            q_data,
            ee_data,
            cpoint_data,
            goal_data,
            obs_data,
            save_path,
        )


def make_2d_animation(
    t_data: list[float],
    joint_data: list[list[list[float]]],
    epoch_max, step,
    q_data=None,
    ee_data=None,
    cpoint_data=None,
    goal_data=None,
    obs_data=None,
    save_path: Union[str, None]=None,
):
    TASK_DIM = 2

    ### 描写範囲を決定 ##

    all_data = []
    for i in range(TASK_DIM):
        temp = []
        if goal_data is not None:
            temp.append(goal_data[i, 0])
        if obs_data is not None:
            temp.extend(obs_data[:, i].tolist())
        for j in range(epoch_max):
            temp.extend(joint_data[j][i])
        all_data.append(temp)
    
    limits = calc_scale(
        min_x=min(all_data[0]),
        max_x=max(all_data[0]),
        min_y=min(all_data[1]),
        max_y=max(all_data[1]),
    )
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    time_template = 'time = %.2f [s]'
    
    #print(joint_data)
    
    if obs_data is not None:
        ax.scatter(
            obs_data[:, 0], obs_data[:, 1],
            label = 'obstacle point', marker = '.', color = 'k',
        )
    if goal_data is not None:
        goal_p = ax.scatter(
            goal_data[0, 0], goal_data[0, 1],
            s = 100, label = 'goal point', marker = '*', color = '#ff7f00', 
            alpha = 1, linewidths = 1.5, edgecolors = 'red'
        )
    
    joint_p, = ax.plot(joint_data[0][0], joint_data[0][1], label="joints", marker="o")
    if ee_data is not None:
        ee_p, = ax.plot(ee_data[0, 0], ee_data[1, 0], label="ee")
    if cpoint_data is not None:
        cp_p = ax.scatter(cpoint_data[0][0], cpoint_data[0][1], label="cpoint")
    tx = ax.set_title(time_template % t_data[0])
    
    ax.grid(True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect('equal')
    ax.legend()
    
    #print("step = ", step)
    
    def update(i: int):
        #print(t_data[i])
        joint_p.set_data(joint_data[i][0], joint_data[i][1])
        
        if ee_data is not None:
            ee_p.set_data(ee_data[0, :i*step], ee_data[1, :i*step])
        
        if cpoint_data is not None:
            temp = list(zip(*cpoint_data[i]))
            cp_p.set_offsets(temp)
        
        if goal_data is not None:
            goal_p.set_offsets([goal_data[i, 0], goal_data[i, 1]])

        tx.set_text(time_template % t_data[i])

        return joint_p, ee_p, cp_p, goal_p, tx
    


    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(epoch_max),
        blit=True
    )
    
    if save_path is not None:
        ani.save(save_path, fps=60, writer='pillow')

    return ani




def make_3d_animation(
    t_data: list[float],
    joint_data: list[list[list[float]]],
    epoch_max : int, step,
    q_data=None,
    ee_data=None,
    cpoint_data=None,
    goal_data=None,
    obs_data=None,
    save_path: Union[str, None]=None,
):
    TASK_DIM = 3

    
    ### 描写範囲を決定 ##
    all_data = []
    for i in range(TASK_DIM):
        temp = []
        if goal_data is not None:
            temp.extend(np.ravel(goal_data[:, i]).tolist())
        if obs_data is not None:
            temp.extend(np.ravel(obs_data[:, i]).tolist())
        for j in range(epoch_max):
            temp.extend(joint_data[j][i])
        all_data.append(temp)
    
    limits = calc_scale(
        min_x=min(all_data[0]),
        max_x=max(all_data[0]),
        min_y=min(all_data[1]),
        max_y=max(all_data[1]),
        min_z=min(all_data[2]),
        max_z=max(all_data[2])
    )
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection="3d")
    time_template = 'time = %.2f [s]'
    
    if obs_data is not None:
        ax.scatter(
            obs_data[:, 0], obs_data[:, 1], obs_data[:, 2],
            label = 'obstacle point', marker = '.', color = 'k',
        )
    if goal_data is not None:
        goal_p = ax.scatter(
            goal_data[0, 0], goal_data[0, 1], goal_data[0, 2],
            s = 100, label = 'goal point', marker = '*', color = '#ff7f00', 
            alpha = 1, linewidths = 1.5, edgecolors = 'red'
        )
    
    joint_p, = ax.plot(
        joint_data[0][0], joint_data[0][1], joint_data[0][2],
        label="joints", marker="o", c="m", alpha=0.5
    )
    if ee_data is not None:
        ee_p, = ax.plot(ee_data[0, 0], ee_data[1, 0], ee_data[2, 0], label="ee", c='#ff7f00')
    if cpoint_data is not None:
        cp_p = ax.scatter(cpoint_data[0][0], cpoint_data[0][1], cpoint_data[0][2], label="cpoint")
    tx = ax.set_title(time_template % t_data[0])
    
    ax.grid(True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_zlim(limits[4], limits[5])
    ax.set_box_aspect((1,1,1)) #あとでかえる
    ax.legend()
    
    def update(i: int):
        joint_p.set_data(joint_data[i][0], joint_data[i][1])
        joint_p.set_3d_properties(joint_data[i][2])
        
        if ee_data is not None:
            ee_p.set_data(ee_data[0, :i], ee_data[1, :i])
            ee_p.set_3d_properties(ee_data[2, :i])
        
        if cpoint_data is not None:
            temp = list(zip(*cpoint_data[i]))
            cp_p._offsets3d = (cpoint_data[i][0], cpoint_data[i][1], cpoint_data[i][2])
        
        if goal_data is not None:
            goal_p._offsets3d = ([goal_data[i, 0]], [goal_data[i, 1]], [goal_data[i, 2]])

        tx.set_text(time_template % t_data[i])

        return joint_p, ee_p, cp_p, goal_p, tx
    


    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(epoch_max),
        #blit=True
    )
    
    if save_path is not None:
        ani.save(save_path, fps=60, writer='pillow')

    return ani








if __name__ == "__main__":
    pass