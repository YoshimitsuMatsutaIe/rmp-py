"""ツリー作成"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np

# ロボットモデルの導入
import robot_baxter.baxter as baxter
import robot_franka_emika.franka_emika as franka_emika
import robot_sice.sice as sice

import mappings
from rmp_leaf import GoalAttractor, JointLimitAvoidance, ObstacleAvoidance
from rmp_node import Node



np.set_printoptions(precision=2)


rmp_param_ex = {
    'jl' : {
        'gamma_p' : 0.1,
        'gamma_d' : 0.05,
        'lam' : 1,
        'sigma' : 0.1
    },
    'attractor' : {
        'max_speed' : 8.0,
        'gain' : 5.0,
        'f_alpha' : 0.15,
        'sigma_alpha' : 1.0,
        'sigma_gamma' : 1.0,
        'wu' : 10.0,
        'wl' : 0.1,
        'alpha' : 0.15,
        'epsilon' : 0.5,
    },
    'obs' : {
        'scale_rep' : 0.2,
        'scale_damp' : 1,
        'gain' : 50,
        'sigma' : 1,
        'rw' : 0.2
    }
}



def multi_solve2(
    node_id: tuple[int, int],
    q, q_dot,g, o_s, rmp_param, robot_name: str
):
    """並列用 : 毎回ノード作成
    """
    
    if robot_name == "baxter":
        robot_model = baxter
    elif robot_name == "franka_emika":
        robot_model = franka_emika
    elif robot_name == "sice":
        robot_model = sice
    else:
        assert False
    
    #print(node_id, end="  ")
    if node_id == (-1, 0):
        node = JointLimitAvoidance(
            name="jl",
            parent=None,
            calc_mappings=mappings.Identity(),
            q_max = robot_model.CPoint.q_max,
            q_min = robot_model.CPoint.q_min,
            q_neutral = robot_model.CPoint.q_neutral,
            parent_dim=robot_model.CPoint.q_neutral.shape[0],
            **rmp_param["joint_limit_avoidance"]
        )
    else:
        temp_map = robot_model.CPoint(*node_id)
        node = Node(
            name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
            dim = robot_model.CPoint.t_dim,
            parent = None,
            mappings = temp_map
        )
        if node_id == robot_model.CPoint.ee_id:
            ### 目標 ###
            g_dot = np.zeros_like(g)
            attracter = GoalAttractor(
                name="ee-attractor",
                parent=node,
                dim=g.shape[0],
                calc_mappings=mappings.Translation(g, g_dot),
                **rmp_param["goal_attractor"]
            )
            node.add_child(attracter)

        ### 障害物 ###
        for i, o in enumerate(o_s):
            obs_node = ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + node.name,
                parent = node,
                calc_mappings = mappings.Distance(o, np.zeros_like(o)),
                **rmp_param["obstacle_avoidance"]
            )
            node.add_child(obs_node)

    node.isMulti = True
    f, M = node.solve(q, q_dot)
    #print(node.name)
    #print("  f = ", f.T)
    #print("  M = ", np.linalg.eigvals(M))
    return f, M



def solve(q, q_dot, g, o_s, robot_name, rmp_param=rmp_param_ex):
    
    if robot_name == 'baxter':
        robot_model = baxter
    elif robot_name == 'franka_emika':
        robot_model = franka_emika
    elif robot_name == "sice":
        robot_model = sice
    else:
        assert False, robot_name + "is not exit"
    
    
    #core=1
    core = cpu_count()-1
    #core = 2
    with Pool(core) as p:
        ### プロセス毎にサブツリーを再構成して計算 ###
        node_ids = [(-1, 0)]
        for i, Rs in enumerate(robot_model.CPoint.R_BARS_ALL):
            node_ids += [(i, j) for j in range(len(Rs))]
        result = p.starmap(
            func = multi_solve2,
            iterable = ((node_id, q, q_dot, g, o_s, rmp_param, robot_name) for node_id in node_ids)
        )
    
    
    f = np.zeros_like(result[0][0])
    M = np.zeros_like(result[0][1])
    for r in result:
        f += r[0]
        M += r[1]
    
    #print(self.f)
    q_ddot = np.linalg.pinv(M) @ f
    
    return q_ddot
