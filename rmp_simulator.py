"""rmpシミュレーション"""

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

from environment import set_point, set_obstacle
import rmp_node
# import rmp_leaf
import tree_constructor
# import mappings
import visualization
from robot_utils import KinematicsAll, get_robot_model, get_cpoint_ids
from planning_ryo import planning

class Simulator:
    
    def __init__(self):
        self.flag = -1
        pass
    
    def print_progress(self, t):
        """sipy用プログレスバー"""
        tmp = int(100 * t / self.TIME_SPAN)
        a, b = divmod(tmp, 10)
        if b == 0 and self.flag != a:
            print(tmp, "%")
        self.flag = a
    
    
    def dx_single(self, t, x, root: rmp_node.Root):
        self.print_progress(t)
        
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


    def set_simulation(
        self,
        param_path: Union[str, None]=None,
        param_dict: Union[dict, None]=None
        ):
        date_now = datetime.datetime.now()
        name = date_now.strftime('%Y-%m-%d--%H-%M-%S')
        self.dir_base = "../rmp_result/rmp-py_result/" + name + "/"
        os.makedirs(self.dir_base, exist_ok=True)
        
        if param_path is not None:
            with open(param_path) as f:
                param = json.load(f)
            shutil.copy2(param_path, self.dir_base)  # 設定ファイルのコピー作成
        elif param_dict is not None:
            json_name = param_dict.pop("json_name")
            with open(self.dir_base + json_name, "w") as f:
                json.dump(param_dict, f)
            param = param_dict
        else:
            assert False
        
        self.robot_name = param["robot_name"]
        self.ex_robot_param = param["ex_robot_param"]
        
        
        self.rm = get_robot_model(param["robot_name"])
        
        c_ = self.rm.CPoint(0, 0, **self.ex_robot_param)
        self.c_dim = c_.c_dim
        self.t_dim = c_.t_dim
        self.ee_id = c_.ee_id
        self.q_neutral = c_.q_neutral
        self.q_max = c_.q_max
        self.q_min = c_.q_min
        self.km = KinematicsAll(self.robot_name, self.ex_robot_param)
        self.calc_joint_position_all = c_.calc_joint_position_all
        
        env = param["env_param"]
        self.obstacle = set_obstacle(env["obstacle"])
        self.goal = set_point(**env["goal"]["param"])[0]

        if 'initial_value' in param:
            print(param["initial_value"])
            assert len(param["initial_value"]) == self.c_dim
            self.q0 = np.array([param["initial_value"]]).T
        else:
            self.q0 = self.q_neutral


        return param


    def run_rmp_sim(
        self,
        param_path: Union[str, None]=None,
        param_dict: Union[dict, None]=None,
        method: str="single"
    ):
        print("running...")
        
        param = self.set_simulation(param_path, param_dict)
        self.rmp_param = param["rmp_param"]

        
        self.node_ids = get_cpoint_ids(self.rm, self.ex_robot_param)
        self.node_ids.append((-1, 0))
        
        self.TIME_SPAN = param["time_span"]
        TIME_INTERVAL = param["time_interval"]
        
        # 初期値
        t_span = (0, self.TIME_SPAN)
        t_eval = np.arange(0, self.TIME_SPAN, TIME_INTERVAL)
        x0 = np.ravel(
            np.concatenate([
                self.q0,
                np.zeros_like(self.q0)
            ])
        )
        
        ### main ###
        t0 = time.perf_counter()
        if method == "single":
            obs_ = self.obstacle
            self.obstacle = np.concatenate(self.obstacle, axis=1)
            root = tree_constructor.make_tree_root(
                self.node_ids, self.goal, self.obstacle, self.rmp_param, self.robot_name,
                self.ex_robot_param, self.c_dim, self.t_dim, self.ee_id,
                self.q_neutral, self.q_max, self.q_min
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
            x0 = np.ravel(self.q_neutral).tolist() + [0 for _ in range(self.c_dim)]
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



        ## CSV保存
        # まずはヘッダーを準備
        header = "t"
        for i in range(self.c_dim):
            header += ",x" + str(i)
        for i in range(self.c_dim):
            header += ",dx" + str(i)
        

        # 時刻歴tと解xを一つのndarrayにする
        data = np.concatenate(
            [sol.t.reshape(1, len(sol.t)).T, sol.y.T],  # sol.tは1次元配列なので2次元化する
            axis=1
        )

        # csvで保存
        np.savetxt(
            self.dir_base + 'configration.csv',
            data,
            header = header,
            comments = '',
            delimiter = ","
        )
        
        ee_ = [self.km.calc_ee_position(data[i:i+1, 1:self.c_dim+1].T) for i in range(len(sol.t))]
        ee_ = np.concatenate(ee_, axis=1)
        error = np.linalg.norm(self.goal - ee_, axis=0)
        error_data = np.stack(
            [sol.t, error]
        ).T
        np.savetxt(
            self.dir_base + 'error.csv',
            error_data,
            header = "t,error",
            comments = '',
            delimiter = ","
        )
        
        ### 以下グラフ化 ###
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 15))
        for i in range(self.c_dim):
            axes[0].plot(sol.t, sol.y[i], label="q" + str(i))
            axes[1].plot(sol.t, sol.y[i+self.c_dim], label="dq" + str(i))
        axes[2].plot(sol.t, error, label="error")
        axes[2].set_ylim(0,)
        
        for i in range(3):
            axes[i].legend()
            axes[i].grid()
        
        axes[2].set_xlabel("time [s]")
        
        fig.savefig(self.dir_base+"configration.png")


        #### 以下アニメ化 ###
        q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
            q_s = [sol.y[i] for i in range(self.c_dim)],
            cpoint_phi_s=self.km.calc_cpoint_position_all,
            joint_phi_s=self.calc_joint_position_all,
            ee_phi=self.km.calc_ee_position,
            is3D=True if self.t_dim==3 else False,
            #ee_phi=rm.o_ee
        )

        if self.t_dim == 3:
            is3D = True
            goal_data = np.array([[self.goal[0,0], self.goal[1,0], self.goal[2,0]]*len(sol.t)]).reshape(len(sol.t), 3)
        elif self.t_dim == 2:
            is3D = False
            goal_data = np.array([[self.goal[0,0], self.goal[1,0],] * len(sol.t)]).reshape(len(sol.t), 2)
        else:
            assert False


        visualization.make_animation(
            t_data = sol.t,
            joint_data=joint_data,
            cpoint_data=cpoint_data,
            ee_data=ee_data,
            is3D=is3D,
            goal_data=goal_data,
            obs_data=np.concatenate(self.obstacle, axis=1).T,
            save_path=self.dir_base+"animation.gif",
            #epoch_max=120
        )
        
        plt.show()


    def run_planning_sim(
        self,
        param_path: Union[str, None]=None,
        param_dict: Union[dict, None]=None,
        method: str="single"
        ):
        param = self.set_simulation(param_path, param_dict)
        pp = param["planning_param"]
        
        q_path_list = planning(
            robot_name=self.robot_name,
            ex_robot_param=self.ex_robot_param,
            q_init = self.q0,
            q_step = pp["q_step"],
            goal = self.goal,
            obs_R = pp["obs_R"],
            obs_Cs = self.obstacle,
            Kat = pp["Kat"],
            Kre = pp["Kre"],
            q_step_n=pp["q_step_n"],
            max_step=1000
        )
        
        data = np.concatenate(q_path_list, axis=1)
        ee_ = [self.km.calc_ee_position(q) for q in q_path_list]
        ee_ = np.concatenate(ee_, axis=1)
        error_data = np.linalg.norm(self.goal - ee_, axis=0)
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        for i in range(self.c_dim):
            axes[0].plot(data[i, :], label="q" + str(i))
        axes[1].plot(error_data, label="error")
        axes[1].set_ylim(0,)
        
        for i in range(2):
            axes[i].legend()
            axes[i].grid()
        
        axes[1].set_xlabel("i")
        
        fig.savefig(self.dir_base+"configration.png")
        
        
        #### 以下アニメ化 ###
        t = list(range(len(q_path_list)))

        q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
            q_s = data.tolist(),
            cpoint_phi_s=self.km.calc_cpoint_position_all,
            joint_phi_s=self.calc_joint_position_all,
            ee_phi=self.km.calc_ee_position,
            is3D=True if self.t_dim==3 else False,
            #ee_phi=rm.o_ee
        )

        if self.t_dim == 3:
            is3D = True
            goal_data = np.array([[self.goal[0,0], self.goal[1,0], self.goal[2,0]]*len(t)]).reshape(len(t), 3)
        elif self.t_dim == 2:
            is3D = False
            goal_data = np.array([[self.goal[0,0], self.goal[1,0],] * len(t)]).reshape(len(t), 2)
        else:
            assert False


        visualization.make_animation(
            t_data = t,
            joint_data=joint_data,
            cpoint_data=cpoint_data,
            ee_data=ee_data,
            is3D=is3D,
            goal_data=goal_data,
            obs_data=np.concatenate(self.obstacle, axis=1).T,
            save_path=self.dir_base+"animation.gif",
            #epoch_max=120
        )
        
        plt.show()


    def environment_visualization(
        self,
        param_path: Union[str, None]=None,
        param_dict: Union[dict, None]=None,
    ):
        """チェック用"""

        param = self.set_simulation(param_path, param_dict)

        self.robot_name = param["robot_name"]
        self.rmp_param = param["rmp_param"]
        env = param["env_param"]
        
        self.obstacle = set_obstacle(env["obstacle"])
        self.goal = set_point(**env["goal"]["param"])[0]
        rm = get_robot_model(param["robot_name"])

        x0 = np.concatenate([
            self.q_neutral,
            self.q_neutral
        ], axis=1)

        t = [0. for _ in range(2)]
        y = x0.tolist()

        q_data, joint_data, ee_data, cpoint_data = visualization.make_data(
            q_s = [y[i] for i in range(self.c_dim)],
            cpoint_phi_s=self.km.calc_cpoint_position_all,
            joint_phi_s=self.calc_joint_position_all,
            ee_phi=self.km.calc_ee_position,
            is3D=True if self.t_dim==3 else False,
            #ee_phi=rm.o_ee
        )


        if self.t_dim == 3:
            is3D = True
            goal_data = np.array([[self.goal[0,0], self.goal[1,0], self.goal[2,0]]*len(t)]).reshape(len(t), 3)
        elif self.t_dim == 2:
            is3D = False
            goal_data = np.array([[self.goal[0,0], self.goal[1,0],] * len(t)]).reshape(len(t), 2)
        else:
            assert False


        visualization.make_animation(
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
