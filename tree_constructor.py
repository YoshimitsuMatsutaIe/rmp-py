"""ツリー作成"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np

# ロボットモデルの導入
import robot_baxter.baxter as baxter
import robot_franka_emika.franka_emika as franka_emika
#import robot_franka_emika_numba.franka_emika as franka_emika
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
    q, q_dot, g, o_s, rmp_param, robot_name: str
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
            q_max = robot_model.q_max(),
            q_min = robot_model.q_min(),
            q_neutral = robot_model.q_neutral(),
            parent_dim=robot_model.CPoint.c_dim,
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
            g_dot = np.zeros(g.shape)
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
                calc_mappings = mappings.Distance(o, np.zeros(o.shape)),
                **rmp_param["obstacle_avoidance"]
            )
            node.add_child(obs_node)

    node.isMulti = True
    f, M = node.solve(q, q_dot)

    return f, M




def multi_solve3(
    node_id: tuple[int, int],
    q: list[float], q_dot: list[float],
    g: list[float],
    o_s: list[list[float]],
    rmp_param, robot_name: str
):
    """並列用 : 毎回ノード作成
    
    データをndarray -> listに変更  
    なぜか遅い  
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
            q_max = robot_model.q_max(),
            q_min = robot_model.q_min(),
            q_neutral = robot_model.q_neutral(),
            parent_dim=robot_model.CPoint.c_dim,
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
            g_dot = np.zeros((len(g), 1))
            attracter = GoalAttractor(
                name="ee-attractor",
                parent=node,
                dim=len(g),
                calc_mappings=mappings.Translation(np.array([g]).T, g_dot),
                **rmp_param["goal_attractor"]
            )
            node.add_child(attracter)

        ### 障害物 ###
        for i, o in enumerate(o_s):
            obs_node = ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + node.name,
                parent = node,
                calc_mappings = mappings.Distance(np.array([o]).T, np.zeros((len(o), 1))),
                **rmp_param["obstacle_avoidance"]
            )
            node.add_child(obs_node)

    node.isMulti = True
    f, M = node.solve(
        np.array([q]).T, np.array([q_dot]).T)

    return f.tolist(), M.tolist()



def solve(q, q_dot, g, o_s, robot_name, node_ids, rmp_param=rmp_param_ex):
    
    
    #core=1
    core = cpu_count()
    #core = 2
    with Pool(core) as p:
        ### プロセス毎にサブツリーを再構成して計算 ###
        result = p.starmap(
            func = multi_solve2,
            iterable = ((node_id, q, q_dot, g, o_s, rmp_param, robot_name) for node_id in node_ids)
        )
    
    
    f = np.zeros(q.shape)
    M = np.zeros((q.shape[0], q.shape[0]))
    for r in result:
        f += r[0]
        M += r[1]
    
    #print(self.f)
    q_ddot = np.linalg.pinv(M) @ f
    
    return q_ddot



def solve3(
    q: list[float], q_dot: list[float],
    g: list[float], o_s: list[list[float]],
    robot_name, node_ids, rmp_param=rmp_param_ex
):
    """ndarrayではなくlistで通信"""
    
    core = cpu_count()
    #core = 2
    with Pool(core) as p:
        ### プロセス毎にサブツリーを再構成して計算 ###
        result = p.starmap(
            func = multi_solve3,
            iterable = ((node_id, q, q_dot, g, o_s, rmp_param, robot_name) for node_id in node_ids)
        )
    
    
    f = np.zeros((len(q), 1))
    M = np.zeros((len(q), len(q)))
    for r in result:
        f += np.array(r[0])
        M += np.array(r[1])
    
    #print(self.f)
    q_ddot = np.linalg.pinv(M) @ f
    
    return q_ddot



def make_node(
    node_id: tuple[int, int],
    g, o_s, rmp_param, robot_name: str
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
            q_max = robot_model.q_max(),
            q_min = robot_model.q_min(),
            q_neutral = robot_model.q_neutral(),
            parent_dim=robot_model.CPoint.c_dim,
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
    
    return node


def make_tree(g, o_s, robot_name, rmp_param, ):
    if robot_name == "baxter":
        robot_model = baxter
    elif robot_name == "franka_emika":
        robot_model = franka_emika
    elif robot_name == "sice":
        robot_model = sice
    else:
        assert False
    
    node_ids = [(-1, 0)]
    for i, Rs in enumerate(robot_model.CPoint.RS_ALL):
        node_ids += [(i, j) for j in range(len(Rs))]
    
    
    nodes = []
    for id in node_ids:
        nodes.append(make_node(id, g, o_s, rmp_param, robot_name))
    
    return nodes


