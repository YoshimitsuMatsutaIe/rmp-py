"""ロボットアームをアニメ化する"""


from turtle import color
import matplotlib.animation as anm
import numpy as np
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
    
    #print(joint_phi_s)
    
    task_dim = 3 if is3D else 2
    T_SIZE = len(q_s[0])  # ループ回数
    
    q = np.array(q_s)  # 縦にくっつくける
    
    joint_data = []
    for i in range(T_SIZE):
        temp = [phi(q[:, i:i+1]) for phi in joint_phi_s]
        #print(temp)
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
            temp = [phi(q[:, i:i+1]) for phi in cpoint_phi_s]
            temp = np.concatenate(temp, axis=1)
            cpoint_data.append(temp.tolist())
    else:
        cpoint_data = None
    
    return q.T, joint_data, ee_data, cpoint_data



# def make_animation_pld(
#     t_data: list[float],
#     joint_data: list[list[list[float]]],
#     q_data=None,
#     ee_data=None,
#     cpoint_data=None,
#     goal_data=None,
#     obs_data=None,
#     is3D: bool=True,
#     epoch_max: int=60,
#     isSave=False,
#     save_path: Union[str, None]=None,
# ):
#     start_time = time.perf_counter()
#     task_dim = 3 if is3D else 2
    
#     T_SIZE = len(t_data)
    
#     ### 描写範囲を決定 ##

#     all_data = []
#     for i in range(task_dim):
#         temp = []
#         if goal_data is not None:
#             temp.append(goal_data[i, 0])
#         if obs_data is not None:
#             temp.extend(obs_data[:, i].tolist())
#         for j in range(T_SIZE):
#             temp.extend(joint_data[j][i])
#         all_data.append(temp)
    
#     limits = calc_scale(
#         min_x=min(all_data[0]),
#         max_x=max(all_data[0]),
#         min_y=min(all_data[1]),
#         max_y=max(all_data[1]),
#         min_z=min(all_data[2]) if is3D else None,
#         max_z=max(all_data[2]) if is3D else None,
#     )
    
    
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(projection="3d") if is3D else fig.add_subplot()
#     time_template = 'time = %.2f [s]'
    

#     def update(i: int):
#         ax.cla()
#         ax.grid(True)
#         ax.set_xlabel('X [m]')
#         ax.set_ylabel('Y [m]')
#         ax.set_xlim(limits[0], limits[1])
#         ax.set_ylim(limits[2], limits[3])
#         if is3D:
#             ax.set_zlabel('Z [m]')
#             assert len(limits) == 6
#             ax.set_zlim(limits[4], limits[5])
            
#         ax.set_box_aspect((1,1,1)) if is3D else ax.set_aspect('equal')
        
#         if is3D:
#             ax.plot(joint_data[i][0], joint_data[i][1], joint_data[i][2], label="joints", marker="o")
#         else:
#             ax.plot(joint_data[i][0], joint_data[i][1], label="joints", marker="o")
        
#         if ee_data is not None:
#             if is3D:
#                 ax.plot(ee_data[:i, 0], ee_data[:i, 1], ee_data[:i, 2], label="ee")
#             else:
#                 ax.plot(ee_data[:i, 0], ee_data[:i, 1], label="ee")
        
#         if cpoint_data is not None:
#             if is3D:
#                 ax.scatter(cpoint_data[i][0], cpoint_data[i][1], cpoint_data[i][2])
#             else:
#                 ax.scatter(cpoint_data[i][0], cpoint_data[i][1])
        
#         if goal_data is not None:
#             if is3D:
#                 ax.scatter(
#                     goal_data[i, 0], goal_data[i, 1], goal_data[i, 2],
#                     s = 100, label = 'goal point', marker = '*', color = '#ff7f00', 
#                     alpha = 1, linewidths = 1.5, edgecolors = 'red'
#                 )
#             else:
#                 ax.scatter(
#                     goal_data[i, 0], goal_data[i, 1],
#                     s = 100, label = 'goal point', marker = '*', color = '#ff7f00', 
#                     alpha = 1, linewidths = 1.5, edgecolors = 'red'
#                 )
        
#         if obs_data is not None:
#             if len(obs_data) == len(t_data):
#                 os = obs_data[i]
#                 os = os.reshape(len(os)//task_dim, task_dim)
#                 if is3D:
#                     ax.scatter(
#                         os[:, 0], os[:, 1], os[:, 2],
#                         label = 'obstacle point', marker = '.', color = 'k',
#                     )
#                 else:
#                     ax.scatter(
#                         os[:, 0], os[:, 1],
#                         label = 'obstacle point', marker = '.', color = 'k',
#                     )
#             else:
#                 if is3D:
#                     ax.scatter(
#                         obs_data[:, 0], obs_data[:, 1], obs_data[:, 2],
#                         label = 'obstacle point', marker = '.', color = 'k',
#                     )
#                 else:
#                     ax.scatter(
#                         obs_data[:, 0], obs_data[:, 1],
#                         label = 'obstacle point', marker = '.', color = 'k',
#                     )
        
#         ax.set_title(time_template % t_data[i])
#         ax.legend()
        
#         return
    
#     if T_SIZE < epoch_max:
#         step = 1
#     else:
#         step = T_SIZE // epoch_max
    
#     print("step = ", step)
#     ani = anm.FuncAnimation(
#         fig = fig,
#         func = update,
#         frames = range(0, len(t_data), step)
#     )
    
#     if isSave:
#         assert save_path is not None
#         ani.save(save_path, fps=60, writer='pillow')
#     # with open(save_dir_path + 'animation.binaryfile', 'wb') as f:
#     #     pickle.dump(ani, f)
    
    
#     print("time = ", time.perf_counter() - start_time)
    
#     #plt.show()
#     return ani



def make_animation(
    t_data: list[float],
    joint_data: list[list[list[float]]],
    q_data=None,
    ee_data=None,
    cpoint_data=None,
    goal_data=None,
    obs_data=None,
    is3D: bool=True,
    epoch_max: int=60,
    save_path: Union[str, None]=None,
):
    if is3D:
        return make_3d_animation(
            t_data,
            joint_data,
            q_data,
            ee_data,
            cpoint_data,
            goal_data,
            obs_data,
            epoch_max,
            save_path
        )
    else:
        return make_2d_animation(
            t_data,
            joint_data,
            q_data,
            ee_data,
            cpoint_data,
            goal_data,
            obs_data,
            epoch_max,
            save_path
        )


def make_2d_animation(
    t_data: list[float],
    joint_data: list[list[list[float]]],
    q_data=None,
    ee_data=None,
    cpoint_data=None,
    goal_data=None,
    obs_data=None,
    epoch_max: int=60,
    save_path: Union[str, None]=None,
):
    start_time = time.perf_counter()
    TASK_DIM = 2
    T_SIZE = len(t_data)
    
    ### 描写範囲を決定 ##

    all_data = []
    for i in range(TASK_DIM):
        temp = []
        if goal_data is not None:
            temp.append(goal_data[i, 0])
        if obs_data is not None:
            temp.extend(obs_data[:, i].tolist())
        for j in range(T_SIZE):
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
        ee_p, = ax.plot(ee_data[:0, 0], ee_data[:0, 1], label="ee")
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
    
    def update(i: int):
        joint_p.set_data(joint_data[i][0], joint_data[i][1])
        
        if ee_data is not None:
            ee_p.set_data(ee_data[:i, 0], ee_data[:i, 1])
        
        if cpoint_data is not None:
            temp = list(zip(*cpoint_data[i]))
            cp_p.set_offsets(temp)
        
        if goal_data is not None:
            goal_p.set_offsets([goal_data[i, 0], goal_data[i, 1]])

        tx.set_text(time_template % t_data[i])

        return joint_p, ee_p, cp_p, goal_p, tx
    
    if T_SIZE < epoch_max:
        step = 1
    else:
        step = T_SIZE // epoch_max
    

    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(t_data), step),
        blit=True
    )
    
    if save_path is not None:
        ani.save(save_path, fps=60, writer='pillow')

    print("time = ", time.perf_counter() - start_time)

    return ani




def make_3d_animation(
    t_data: list[float],
    joint_data: list[list[list[float]]],
    q_data=None,
    ee_data=None,
    cpoint_data=None,
    goal_data=None,
    obs_data=None,
    epoch_max: int=60,
    save_path: Union[str, None]=None,
):
    start_time = time.perf_counter()
    TASK_DIM = 3
    T_SIZE = len(t_data)
    
    ### 描写範囲を決定 ##
    all_data = []
    for i in range(TASK_DIM):
        temp = []
        if goal_data is not None:
            temp.extend(np.ravel(goal_data[:, i]).tolist())
        if obs_data is not None:
            temp.extend(np.ravel(obs_data[:, i]).tolist())
        for j in range(T_SIZE):
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
        ee_p, = ax.plot(ee_data[:0, 0], ee_data[:0, 1], ee_data[:0, 2], label="ee", c='#ff7f00')
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
            ee_p.set_data(ee_data[:i, 0], ee_data[:i, 1])
            ee_p.set_3d_properties(ee_data[:i, 2])
        
        if cpoint_data is not None:
            temp = list(zip(*cpoint_data[i]))
            cp_p._offsets3d = (cpoint_data[i][0], cpoint_data[i][1], cpoint_data[i][2])
        
        if goal_data is not None:
            goal_p._offsets3d = ([goal_data[i, 0]], [goal_data[i, 1]], [goal_data[i, 2]])

        tx.set_text(time_template % t_data[i])

        return joint_p, ee_p, cp_p, goal_p, tx
    
    if T_SIZE < epoch_max:
        step = 1
    else:
        step = T_SIZE // epoch_max
    

    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(t_data), step),
        #blit=True
    )
    
    if save_path is not None:
        ani.save(save_path, fps=60, writer='pillow')

    print("time = ", time.perf_counter() - start_time)

    return ani








if __name__ == "__main__":
    pass