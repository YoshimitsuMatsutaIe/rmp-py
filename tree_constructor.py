"""ツリー作成"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np

# 
import mappings
from rmp_leaf import GoalAttractor
from rmp_leaf import JointLimitAvoidance
from rmp_leaf import ObstacleAvoidance, ObstacleAvoidanceMulti
from rmp_node import Node, Root
from robot_utils import get_robot_model

np.set_printoptions(precision=2)



def multi_solve2(
    node_id: tuple[int, int],
    q, q_dot, g, o_s, rmp_param, robot_name: str, ex_robot_param
):
    """並列用 : 毎回ノード作成
    """

    rm = get_robot_model(robot_name)
    rm_ = rm.CPoint(0,0,**ex_robot_param)
    
    
    #print(node_id, end="  ")
    if node_id == (-1, 0):
        node = JointLimitAvoidance(
            name="jl",
            calc_mappings=mappings.Identity(),
            q_max = rm_.q_max,
            q_min = rm_.q_min,
            q_neutral = rm_.q_neutral,
            parent_dim=rm_.c_dim,
            **rmp_param["joint_limit_avoidance"]
        )
    else:
        temp_map = rm.CPoint(*node_id, **ex_robot_param)
        node = Node(
            name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
            dim = rm_.t_dim,
            mappings = temp_map
        )
        if node_id == rm_.ee_id:
            ### 目標 ###
            g_dot = np.zeros(g.shape)
            attracter = GoalAttractor(
                name="ee-attractor",
                dim=g.shape[0],
                calc_mappings=mappings.Translation(g, g_dot),
                **rmp_param["goal_attractor"]
            )
            node.add_child(attracter)

        ### 障害物 ###
        for i, o in enumerate(o_s):
            obs_node = ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + node.name,
                calc_mappings = mappings.Distance(o, np.zeros(o.shape)),
                **rmp_param["obstacle_avoidance"],
                parent_dim=rm_.t_dim
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
    rmp_param, robot_name: str, ex_robot_param
):
    """並列用 : 毎回ノード作成
    
    データをndarray -> listに変更  
    なぜか遅い  
    """
    
    rm = get_robot_model(robot_name)
    rm_ = rm.CPoint(0, 0, **ex_robot_param)
    
    #print(node_id, end="  ")
    if node_id == (-1, 0):
        node = JointLimitAvoidance(
            name="jl",
            calc_mappings=mappings.Identity(),
            q_max = rm_.q_max,
            q_min = rm_.q_min,
            q_neutral = rm_.q_neutral,
            parent_dim=rm_.c_dim,
            **rmp_param["joint_limit_avoidance"]
        )
    else:
        temp_map = rm.CPoint(*node_id, **ex_robot_param)
        node = Node(
            name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
            dim = rm_.t_dim,
            mappings = temp_map
        )
        if node_id == rm_.ee_id:
            ### 目標 ###
            g_dot = np.zeros((len(g), 1))
            attracter = GoalAttractor(
                name="ee-attractor",
                dim=len(g),
                calc_mappings=mappings.Translation(np.array([g]).T, g_dot),
                **rmp_param["goal_attractor"],
                parent_dim=rm_.t_dim
            )
            node.add_child(attracter)

        ### 障害物 ###
        for i, o in enumerate(o_s):
            obs_node = ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + node.name,
                calc_mappings = mappings.Distance(np.array([o]).T, np.zeros((len(o), 1))),
                **rmp_param["obstacle_avoidance"],
                parent_dim=rm_.t_dim
            )
            node.add_child(obs_node)

    node.isMulti = True
    f, M = node.solve(
        np.array([q]).T, np.array([q_dot]).T)

    return f.tolist(), M.tolist()






def multi_solve4(
    node_id: tuple[int, int],
    q, q_dot, g, o_s, rmp_param, robot_name: str, ex_robot_param
):
    """並列用 : 毎回ノード作成
    """
    
    rm = get_robot_model(robot_name)
    rm_ = rm.CPoint(0, 0, **ex_robot_param)
    
    #print(node_id, end="  ")
    if node_id == (-1, 0):
        node = JointLimitAvoidance(
            name="jl",
            calc_mappings=mappings.Identity(),
            q_max = rm_.q_max,
            q_min = rm_.q_min,
            q_neutral = rm_.q_neutral,
            parent_dim=rm_.c_dim,
            **rmp_param["joint_limit_avoidance"]
        )
    else:
        temp_map = rm.CPoint(*node_id, **ex_robot_param)
        node = Node(
            name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
            dim = rm_.t_dim,
            mappings = temp_map,
            parent_dim = rm_.c_dim
        )
        if node_id == rm_.ee_id:
            ### 目標 ###
            g_dot = np.zeros(g.shape)
            attracter = GoalAttractor(
                name="ee-attractor",
                dim=g.shape[0],
                calc_mappings=mappings.Translation(g, g_dot),
                **rmp_param["goal_attractor"],
                parent_dim=rm_.t_dim
            )
            node.add_child(attracter)

        ### 障害物 ###
        obs_node = ObstacleAvoidanceMulti(
            name="obs_multi" + node.name,
            calc_mappings = mappings.Identity(),
            dim = rm_.t_dim,
            o_s = o_s,
            **rmp_param["obstacle_avoidance"],
            parent_dim=rm_.t_dim
        )
        node.add_child(obs_node)

    node.isMulti = True
    f, M = node.solve(q, q_dot)

    return f, M




def solve(q, q_dot, g, o_s, robot_name, node_ids, rmp_param):
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
    
    return np.linalg.pinv(M) @ f



def solve3(
    q: list[float], q_dot: list[float],
    g: list[float], o_s: list[list[float]],
    robot_name, node_ids, rmp_param
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


def solve4(q, q_dot, g, o_s, robot_name, node_ids, rmp_param):
    #core=1
    core = cpu_count()
    #core = 2
    with Pool(core) as p:
        ### プロセス毎にサブツリーを再構成して計算 ###
        result = p.starmap(
            func = multi_solve4,
            iterable = ((node_id, q, q_dot, g, o_s, rmp_param, robot_name) for node_id in node_ids)
        )
    
    f = np.zeros(q.shape)
    M = np.zeros((q.shape[0], q.shape[0]))
    for r in result:
        f += r[0]
        M += r[1]
    
    return np.linalg.pinv(M) @ f




def make_node(
    node_id: tuple[int, int],
    g, o_s, rmp_param, robot_name: str, ex_robot_param
):
    """並列用 : 毎回ノード作成
    """
    
    rm = get_robot_model(robot_name)
    rm_ = rm.CPoint(0, 0, **ex_robot_param)
    
    #print(node_id, end="  ")
    if node_id == (-1, 0):
        node = JointLimitAvoidance(
            name="jl",
            calc_mappings=mappings.Identity(),
            q_max = rm_.q_max,
            q_min = rm_.q_min,
            q_neutral = rm_.q_neutral,
            parent_dim=rm_.c_dim,
            **rmp_param["joint_limit_avoidance"]
        )
    else:
        temp_map = rm.CPoint(*node_id, **ex_robot_param)
        node = Node(
            name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
            dim = rm_.t_dim,
            mappings = temp_map,
            parent_dim = rm_.c_dim
        )
        if node_id == rm_.ee_id:
            ### 目標 ###
            g_dot = np.zeros_like(g)
            attracter = GoalAttractor(
                name="ee-attractor",
                dim=g.shape[0],
                calc_mappings=mappings.Translation(g, g_dot),
                **rmp_param["goal_attractor"],
                parent_dim=rm_.t_dim
            )
            node.add_child(attracter)

        ### 障害物 ###
        for i, o in enumerate(o_s):
            obs_node = ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + node.name,
                calc_mappings = mappings.Distance(o, np.zeros_like(o)),
                **rmp_param["obstacle_avoidance"],
                parent_dim=rm_.t_dim
            )
            node.add_child(obs_node)
    
    node.isMulti = True
    
    return node


def make_tree(g, o_s, robot_name, rmp_param, ex_robot_param):

    rm = get_robot_model(robot_name)
    rm_ = rm.CPoint(0, 0, **ex_robot_param)
    
    node_ids = [(-1, 0)]
    for i, Rs in enumerate(rm_.RS_ALL):
        node_ids += [(i, j) for j in range(len(Rs))]
    
    
    nodes = []
    for id in node_ids:
        nodes.append(make_node(id, g, o_s, rmp_param, robot_name, ex_robot_param))
    
    return nodes



def make_tree_root(
    node_ids: list[tuple[int, int]],
    g, o_s, rmp_param,
    robot_name: str,
    ex_robot_param,
    c_dim: int, t_dim: int, ee_id,
    q_neutral, q_max, q_min
):
    """single"""

    rm = get_robot_model(robot_name)
    root = Root(c_dim)
    
    for node_id in node_ids:
        if node_id == (-1, 0):
            node = JointLimitAvoidance(
                name="jl",
                calc_mappings=mappings.Identity(),
                q_max = q_max,
                q_min = q_min,
                q_neutral = q_neutral,
                parent_dim=c_dim,
                **rmp_param["joint_limit_avoidance"]
            )
            root.add_child(node)
        else:
            temp_map = rm.CPoint(*node_id, **ex_robot_param)
            node = Node(
                name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
                dim = t_dim,
                mappings = temp_map,
                parent_dim = c_dim
            )
            if node_id == ee_id:
                ### 目標 ###
                g_dot = np.zeros_like(g)
                attracter = GoalAttractor(
                    name="ee-attractor",
                    dim=g.shape[0],
                    calc_mappings=mappings.Translation(g, g_dot),
                    **rmp_param["goal_attractor"],
                    parent_dim=t_dim
                )
                node.add_child(attracter)

            ### 障害物 ###
            obs_node = ObstacleAvoidanceMulti(
                name="obs_multi" + node.name,
                calc_mappings = mappings.Identity(),
                dim = t_dim,
                o_s = o_s,
                **rmp_param["obstacle_avoidance"],
                parent_dim=t_dim
            )
            node.add_child(obs_node)
            root.add_child(node)

    return root


if __name__ == "__main__":
    pass