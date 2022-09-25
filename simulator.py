"""シミュレーション"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#from typing import Union
import datetime
import time
import os
#from pathlib import Path
import shutil
import json
from typing import Union

from environment import *
import rmp_node
# import rmp_leaf
import tree_constructor
# import mappings
import visualization
from robot_utils import get_robot_model


class Simulator:
    
    def __init__(self):
        pass
    
    
    def set_obstacle(self, obs_params: list[dict]):
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
            else:
                assert False
        return obstacle
    
    
    def dx_single(self, t, x, root: rmp_node.Root):
        dim = x.shape[0] // 2
        x = x.reshape(-1, 1)
        q_ddot = root.solve(x[:dim, :], x[dim:, :])
        x_dot = np.concatenate([x[dim:, :], q_ddot])
        return np.ravel(x_dot)
    
    
    def dx(self, t, x):
        """ODE"""
        dim = x.shape[0] // 2
        x = x.reshape(-1, 1)
        q_ddot = tree_constructor.solve(
            q=x[:dim, :], q_dot=x[dim:, :], g=self.goal, o_s=self.obstacle,
            node_ids=self.node_ids,
            robot_name=self.robot_name,
            rmp_param=self.rmp_param
        )

        x_dot = np.concatenate([x[dim:, :], q_ddot])
        return np.ravel(x_dot)

    
    def dx3(self, t, x):
        """ODE list"""
        dim = len(x) // 2
        q_ddot = tree_constructor.solve3(
            q=x[:dim].tolist(),
            q_dot=x[dim:].tolist(),
            g=self.goal_list,
            o_s=self.obstacle_list,
            node_ids=self.node_ids,
            robot_name=self.robot_name,
            rmp_param=self.rmp_param
        )
        x_dot = x[dim:].tolist()
        x_dot.extend(np.ravel(q_ddot).tolist())
        
        return x_dot


    def dx4(self, t, x):
        """ODE 最速"""
        dim = x.shape[0] // 2
        x = x.reshape(-1, 1)
        q_ddot = tree_constructor.solve4(
            q=x[:dim, :], q_dot=x[dim:, :], g=self.goal, o_s=self.obstacle,
            node_ids=self.node_ids,
            robot_name=self.robot_name,
            rmp_param=self.rmp_param
        )
        
        x_dot = np.concatenate([x[dim:, :], q_ddot])
        return np.ravel(x_dot)



    def __make_ndarry_to_list(self,):
        self.goal_list = np.ravel(self.goal).tolist()
        self.obstacle_list = [
            np.ravel(o).tolist() for o in self.obstacle
        ]


    def main(
        self,
        param_path: Union[str, None]=None,
        param_dict: Union[dict, None]=None,
        method: str="single"):
        
        date_now = datetime.datetime.now()
        name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
        base = "../rmp_result/rmp-py_result/" + name + "/"
        os.makedirs(base, exist_ok=True)
        
        if param_path is not None:
            with open(param_path) as f:
                param = json.load(f)
            shutil.copy2(param_path, base)  # 設定ファイルのコピー作成
        elif param_dict is not None:
            json_name = param_dict.pop("json_name")
            with open(base + json_name, "w") as f:
                json.dump(param_dict, f)
            param = param_dict
        else:
            assert False
        
        self.robot_name = param["robot_name"]
        self.rmp_param = param["rmp_param"]
        env = param["env_param"]
        
        self.obstacle = self.set_obstacle(env["obstacle"])
        self.goal = set_point(**env["goal"]["param"])[0]
        rm = get_robot_model(param["robot_name"])
        
        self.node_ids = [(-1, 0)]
        for i, Rs in enumerate(rm.CPoint.RS_ALL):
            self.node_ids += [(i, j) for j in range(len(Rs))]
        
        
        TIME_SPAN = param["time_span"]
        TIME_INTERVAL = param["time_interval"]
        # 初期値
        t_span = (0, TIME_SPAN)
        t_eval = np.arange(0, TIME_SPAN, TIME_INTERVAL)
        x0 = np.ravel(
            np.concatenate([
                rm.q_neutral(),
                np.zeros_like(rm.q_neutral())
            ])
        )
        
        ### main ###
        t0 = time.perf_counter()
        if method == "single":
            obs_ = self.obstacle
            self.obstacle = np.concatenate(self.obstacle, axis=1)
            root = tree_constructor.make_tree_root(
                self.node_ids, self.goal, self.obstacle, self.rmp_param, self.robot_name
            )
            sol = integrate.solve_ivp(
                fun = self.dx_single,
                t_span = t_span,
                y0 = x0,
                t_eval=t_eval,
                args=(root,)
            )
            self.obstacle = obs_
        elif method == "multi_0":
            sol = integrate.solve_ivp(
                fun = self.dx,
                t_span = t_span,
                y0 = x0,
                t_eval=t_eval
            )
        elif method == "multi_3":
            x0 = np.ravel(rm.q_neutral()).tolist() + [0 for _ in range(rm.CPoint.c_dim)]
            self.__make_ndarry_to_list()
            sol = integrate.solve_ivp(
                fun = self.dx3,
                t_span=t_span,
                y0 = x0,
                t_eval=t_eval,
            )
        elif method == "multi_4":
            obs_ = self.obstacle
            self.obstacle = np.concatenate(self.obstacle, axis=1)
            sol = integrate.solve_ivp(
                fun = self.dx4,
                t_span=t_span,
                y0 = x0,
                t_eval=t_eval,
            )
            self.obstacle = obs_
        else:
            assert False
            
        sim_time =  time.perf_counter() - t0
        print("sim time = ", sim_time)
        print(sol.message)



        c_dim = rm.CPoint.c_dim
        t_dim = rm.CPoint.t_dim
        ## CSV保存
        # まずはヘッダーを準備
        header = "t"
        for i in range(c_dim):
            header += ",x" + str(i)
        for i in range(c_dim):
            header += ",dx" + str(i)
        

        # 時刻歴tと解xを一つのndarrayにする
        data = np.concatenate(
            [sol.t.reshape(1, len(sol.t)).T, sol.y.T],  # sol.tは1次元配列なので2次元化する
            axis=1
        )

        # csvで保存
        np.savetxt(
            base + 'configration.csv',
            data,
            header = header,
            comments = '',
            delimiter = ","
        )
        
        ee_map_ = rm.CPoint(*rm.CPoint.ee_id)
        ee_ = [ee_map_.phi(data[i:i+1, 1:c_dim+1].T) for i in range(len(sol.t))]
        ee_ = np.concatenate(ee_, axis=1)
        error = np.linalg.norm(self.goal - ee_, axis=0)
        error_data = np.stack(
            [sol.t, error]
        ).T
        np.savetxt(
            base + 'error.csv',
            error_data,
            header = "t,error",
            comments = '',
            delimiter = ","
        )
        
        
        
        ### 以下グラフ化 ###
        
        
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 15))
        for i in range(c_dim):
            axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
            axes[1].plot(sol.t, sol.y[i+c_dim], label="dq" + str(i))
        axes[2].plot(sol.t, error, label="error")
        axes[2].set_ylim(0,)
        
        for i in range(3):
            axes[i].legend()
            axes[i].grid()
        
        axes[2].set_xlabel("time [s]")
        
        fig.savefig(base+"configration.png")



        #### 以下アニメ化 ###

        cpoint_phis = []
        for i, rs in enumerate(rm.CPoint.RS_ALL):
            for j, _ in enumerate(rs):
                map_ = rm.CPoint(i, j)
                cpoint_phis.append(map_.phi)

        # map_ = rm.CPoint(c_dim, 0)
        # cpoint_phis.append(map_.phi)

        q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
            q_s = [sol.y[i] for i in range(c_dim)],
            cpoint_phi_s=cpoint_phis,
            joint_phi_s=rm.JOINT_PHI(),
            ee_phi=ee_map_.phi,
            is3D=True if t_dim==3 else False,
            #ee_phi=rm.o_ee
        )

        if t_dim == 3:
            is3D = True
            goal_data = np.array([[self.goal[0,0], self.goal[1,0], self.goal[2,0]]*len(sol.t)]).reshape(len(sol.t), 3)
        elif t_dim == 2:
            is3D = False
            goal_data = np.array([[self.goal[0,0], self.goal[1,0],] * len(sol.t)]).reshape(len(sol.t), 2)
        else:
            assert False


        ani = visualization.make_animation(
            t_data = sol.t,
            joint_data=joint_data,
            cpoint_data=cpoint_data,
            ee_data=ee_data,
            is3D=is3D,
            goal_data=goal_data,
            obs_data=np.concatenate(self.obstacle, axis=1).T,
            save_path=base+"animation.gif",
            #epoch_max=120
        )
        
        plt.show()


    def environment_visualization(
        self,
        param_path: Union[str, None]=None,
        param_dict: Union[dict, None]=None,
    ):
        """チェック用"""

        if param_path is not None:
            with open(param_path) as f:
                param = json.load(f)
        elif param_dict is not None:
            param = param_dict
        else:
            assert False
        
        self.robot_name = param["robot_name"]
        self.rmp_param = param["rmp_param"]
        env = param["env_param"]
        
        self.obstacle = self.set_obstacle(env["obstacle"])
        self.goal = set_point(**env["goal"]["param"])[0]
        rm = get_robot_model(param["robot_name"])

        x0 = np.concatenate([
            rm.q_neutral(),
            rm.q_neutral()
        ], axis=1)

        c_dim = rm.CPoint.c_dim
        t_dim = rm.CPoint.t_dim

        t = [0. for _ in range(2)]
        y = x0.tolist()

        cpoint_phis = []
        for i, rs in enumerate(rm.CPoint.RS_ALL):
            for j, _ in enumerate(rs):
                map_ = rm.CPoint(i, j)
                cpoint_phis.append(map_.phi)

        q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
            q_s = [y[i] for i in range(c_dim)],
            cpoint_phi_s=cpoint_phis,
            joint_phi_s=rm.JOINT_PHI(),
            is3D=True if t_dim==3 else False,
            #ee_phi=rm.o_ee
        )

        if t_dim == 3:
            is3D = True
            goal_data = np.array([[self.goal[0,0], self.goal[1,0], self.goal[2,0]]*len(t)]).reshape(len(t), 3)
        elif t_dim == 2:
            is3D = False
            goal_data = np.array([[self.goal[0,0], self.goal[1,0],] * len(t)]).reshape(len(t), 2)
        else:
            assert False


        ani = visualization.make_animation(
            t_data = t,
            joint_data=joint_data,
            cpoint_data=cpoint_data,
            ee_data=ee_data,
            is3D=is3D,
            goal_data=goal_data,
            obs_data=np.concatenate(self.obstacle, axis=1).T,
            #epoch_max=120
        )
        
        plt.show()


if __name__ == "__main__":
    pass
