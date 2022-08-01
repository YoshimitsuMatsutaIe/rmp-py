"""ロボットアームをアニメ化する"""


import matplotlib.animation as anm
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import axes
import time
from functools import lru_cache  # これつけるとplt.show()でアニメーションがループしなくなる
import pickle
from typing import Union




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
    joint_phi_s,
    is3D: bool,
    ee_phi=None,
    cpoint_phi_s=None,
):
    """描写したい点列を生成
    """
    task_dim = 3 if is3D else 2
    T_SIZE = len(q_s[0])  # ループ回数
    
    q = np.array(q_s)  # 縦にくっつくける
    
    joint_data = []
    for i in range(T_SIZE):
        q_ = q[:, i:i+1]
        temp = []
        for phi in joint_phi_s:
            temp.append(phi(q_))
        temp = np.concatenate(temp, axis=1)
        joint_data.append(temp.tolist())

    if ee_phi is not None:
        ee_data = np.empty((T_SIZE, task_dim))
        for i in range(T_SIZE):
            ee_data[i:i+1, :] = ee_phi(q[:, i:i+1]).T
    else:
        ee_data = None

    
    if cpoint_phi_s is not None:
        cpoint_data = []
        for i in range(T_SIZE):
            q_ = q[:, i:i+1]
            temp = []
            for phi in cpoint_phi_s:
                temp.append(phi(q_))
            temp = np.concatenate(temp, axis=1)
            cpoint_data.append(temp.tolist())
    else:
        cpoint_data = None
    
    return q.T, joint_data, ee_data, cpoint_data



def make_animation(
    t_data: list[float],
    joint_data: list[list[list[float]]],
    q_data: Union[NDArray[np.float64], None]=None,
    ee_data: Union[NDArray[np.float64], None]=None,
    cpoint_data: Union[list[list[list[float]]], None]=None,
    goal_data=None,
    obs_data=None,
    is3D: bool=True,
    epoch_max: int=60,
    isSave=False,
    save_path: Union[str, None]=None,
):
    start_time = time.time()
    task_dim = 3 if is3D else 2
    
    T_SIZE = len(t_data)
    
    ### 描写範囲を決定 ##

    all_data = []
    for i in range(task_dim):
        temp = []
        for j in range(T_SIZE):
            temp.extend(joint_data[j][i])
        all_data.append(temp)
    
    limits = calc_scale(
        min_x=min(all_data[0]),
        max_x=max(all_data[0]),
        min_y=min(all_data[1]),
        max_y=max(all_data[1]),
        min_z=min(all_data[2]) if is3D else None,
        max_z=max(all_data[2]) if is3D else None,
    )
    
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection="3d") if is3D else fig.add_subplot()
    time_template = 'time = %.2f [s]'
    

    def update(i: int):
        ax.cla()
        ax.grid(True)
        ax.set_xlabel('X[m]')
        ax.set_ylabel('Y[m]')
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        if is3D:
            ax.set_zlabel('Z[m]')
            assert len(limits) == 6
            ax.set_zlim(limits[4], limits[5])
            
        ax.set_box_aspect((1,1,1)) if is3D else ax.set_aspect('equal')
        
        if is3D:
            ax.plot(joint_data[i][0], joint_data[i][1], joint_data[i][2], label="joints", marker="o")
        else:
            ax.plot(joint_data[i][0], joint_data[i][1], label="joints", marker="o")
        
        if ee_data is not None:
            if is3D:
                ax.scatter(ee_data[i, 0], ee_data[i, 1], ee_data[i, 2], lanel="ee")
            else:
                ax.scatter(ee_data[i, 0], ee_data[i, 1], lanel="ee")
        
        if cpoint_data is not None:
            if is3D:
                ax.scatter(cpoint_data[i][0], cpoint_data[i][1], cpoint_data[i][2])
            else:
                ax.scatter(cpoint_data[i][0], cpoint_data[i][1])
        
        if goal_data is not None:
            if is3D:
                ax.scatter(
                    goal_data[i, 0], goal_data[i, 1], goal_data[i, 2],
                    s = 100, label = 'goal point', marker = '*', color = '#ff7f00', 
                    alpha = 1, linewidths = 1.5, edgecolors = 'red'
                )
            else:
                ax.scatter(
                    goal_data[i, 0], goal_data[i, 1],
                    s = 100, label = 'goal point', marker = '*', color = '#ff7f00', 
                    alpha = 1, linewidths = 1.5, edgecolors = 'red'
                )
        
        if obs_data is not None:
            if len(obs_data) == len(t_data):
                os = obs_data[i]
                os = os.reshape(len(os)//task_dim, task_dim)
                if is3D:
                    ax.scatter(
                        os[:, 0], os[:, 1], os[:, 2],
                        label = 'obstacle point', marker = '.', color = 'k',
                    )
                else:
                    ax.scatter(
                        os[:, 0], os[:, 1],
                        label = 'obstacle point', marker = '.', color = 'k',
                    )
            else:
                if is3D:
                    ax.scatter(
                        obs_data[:, 0], obs_data[:, 1], obs_data[:, 2],
                        label = 'obstacle point', marker = '.', color = 'k',
                    )
                else:
                    ax.scatter(
                        obs_data[:, 0], obs_data[:, 1],
                        label = 'obstacle point', marker = '.', color = 'k',
                    )
        
        ax.set_title(time_template % t_data[i])
        ax.legend()
        
        return
    
    if T_SIZE < epoch_max:
        step = 1
    else:
        step = T_SIZE // epoch_max
    
    print("step = ", step)
    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(t_data), step)
    )
    
    if isSave:
        assert save_path is not None
        ani.save(save_path, fps=60, writer='pillow')
    # with open(save_dir_path + 'animation.binaryfile', 'wb') as f:
    #     pickle.dump(ani, f)
    
    
    print("time = ", time.time() - start_time)
    
    #plt.show()
    return ani





if __name__ == "__main__":
    pass