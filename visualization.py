"""ロボットアームをアニメ化する"""


import matplotlib.animation as anm
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import lru_cache  # これつけるとplt.show()でアニメーションがループしなくなる
import pickle


def calc_scale(max_x, min_x, max_y, min_y, max_z=None, min_z=None):
    """軸範囲を計算"""
    assert max_x > min_x, "hoeg"
    assert max_y > min_y, "hoeg"
    if max_z is not None and min_z is not None:
        assert max_z > min_z, "hoeg"
    
    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    if max_z is None and min_z is None:
        max_range = max(max_x-min_x, max_y-min_y) * 0.5
    else:
        max_range = max(max_x-min_x, max_y-min_y, max_z-min_z) * 0.5
        mid_z = (max_z + min_z) * 0.5
    
    
    if max_z is None and min_z is None:
        return (
            mid_x - max_range, mid_x + max_range,
            mid_y - max_range, mid_y + max_range
        )
    else:
        return (
            mid_x - max_range, mid_x + max_range,
            mid_y - max_range, mid_y + max_range,
            mid_z - max_range, mid_z + max_range
        )



def make_data(
    q_s, joint_phi_s, is3D=True, ee_phi=None, cpoint_phi_s=None,
):
    """描写したい点列を生成
    """
    task_dim = 3 if is3D else 2
    T_SIZE = len(q_s[0])  # 回数
    N_JOINT = len(joint_phi_s)
    
    q = np.array(q_s)  # 縦にくっつくける
    
    joint_data = np.empty((T_SIZE, task_dim*N_JOINT))
    for i in range(T_SIZE):
        temp_q = q[:, i:i+1]
        for j, phi in enumerate(joint_phi_s):
            joint_data[i:i+1, task_dim*j:task_dim*(j+1)] = phi(temp_q).T
    

    if ee_phi is not None:
        ee_data = np.empty((T_SIZE, task_dim))
        for i in range(T_SIZE):
            ee_data[i:i+1, :] = ee_phi(q[:, i:i+1]).T
    else:
        ee_data = None

    
    if cpoint_phi_s is not None:
        cpoint_data = []
        for n in range(N_JOINT):
            if len(cpoint_phi_s[n]) == 0:
                cpoint_data.append(None)
            else:
                cpoint_data.append(
                    np.empty((T_SIZE, task_dim*len(cpoint_phi_s[n])))
                )
        
        for i in T_SIZE:
            temp_q = q[:, i:i+1]
            for j, phis in enumerate(cpoint_phi_s):
                if cpoint_data[j] is not None:
                    for k, phi in enumerate(phis):
                        cpoint_data[j][i:i+1, task_dim*k:task_dim*(k+1)] = phi(temp_q).T
    
    else:
        cpoint_data = None
    
    return q.T, joint_data, ee_data, cpoint_data



def make_animation(
    t_data, joint_data,
    q_data=None, ee_data=None, cpoint_data=None,
    goal_data=None, obs_data=None,
    is3D=True,
    epoch_max=100,
    save_dir_path='',
):
    start_time = time.time()
    
    assert len(t_data) == len(joint_data), "データ数があってない"
    

    task_dim = 3 if is3D else 2
    
    T_SIZE = len(t_data)
    N_JOINT = joint_data.shape[1] // task_dim
    
    # 描写範囲を決定
    
    all_data = joint_data.reshape(T_SIZE*N_JOINT, task_dim)
    
    limits = calc_scale(
        min_x=all_data[:, 0:1].min(),
        max_x=all_data[:, 0:1].max(),
        min_y=all_data[:, 1:2].min(),
        max_y=all_data[:, 1:2].max(),
        min_z=all_data[:, 2:3].min() if is3D else None,
        max_z=all_data[:, 2:3].max() if is3D else None,
    )
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d") if is3D else fig.add_subplot()
    time_template = 'time = %.2f [s]'
    
    
    @lru_cache()
    def update(i):
        ax.cla()
        ax.grid(True)
        ax.set_xlabel('X[m]')
        ax.set_ylabel('Y[m]')
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        if is3D:
            ax.set_zlabel('Z[m]')
            ax.set_zlim(limits[4], limits[5])
            
        ax.set_box_aspect((1,1,1)) if is3D else ax.set_aspect('equal')
        
        js = joint_data[i:i+1, :].reshape(N_JOINT, task_dim)
        if is3D:
            ax.plot(js[:, 0], js[:, 1], js[:, 2], label="joints", marker="o")
        else:
            ax.plot(js[:, 0], js[:, 1], label="joints", marker="o")
        
        if ee_data is not None:
            if is3D:
                ax.scatter(ee_data[i, 0], ee_data[i, 1], ee_data[i, 2], lanel="ee")
            else:
                ax.scatter(ee_data[i, 0], ee_data[i, 1], lanel="ee")
        
        if cpoint_data is not None:
            xc = []
            for m, cs in enumerate(cpoint_data):
                if cs is not None:
                    c = cs[i:i+1, :].reshape(cs.shape[1]//task_dim, task_dim)
                    if is3D:
                        ax.scatter(c[:, 0], c[:, 1], c[:, 2], label=str(m))
                    else:
                        ax.scatter(c[:, 0], c[:, 1], label=str(m))
        
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
        
        ax.set_title(time_template % t_data[i])
        ax.legend()
        
        return
    
    
    if T_SIZE < epoch_max:
        step = 1
    else:
        step = T_SIZE // epoch_max
    
    ani = anm.FuncAnimation(
        fig = fig,
        func = update,
        frames = range(0, len(t_data), step)
    )
    
    ani.save(save_dir_path + "animation.gif", fps=30, writer='pillow')
    # with open(save_dir_path + 'animation.binaryfile', 'wb') as f:
    #     pickle.dump(ani, f)
    
    
    print("time = ", time.time() - start_time)
    
    #plt.show()
    return ani





if __name__ == "__main__":
    pass