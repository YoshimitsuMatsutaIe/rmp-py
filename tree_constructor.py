import numpy as np
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count
import time

import baxter
import mappings
from rmp_node import Node
from rmp_leaf import GoalAttractor, ObstacleAvoidance, JointLimitAvoidance

np.set_printoptions(precision=2)


def multi_solve2(
    node_id: tuple[int, int],
    q, q_dot,
    g: NDArray[np.float64],
    o_s: list[NDArray[np.float64]]
):
    """並列用 : 毎回ノード作成
    """
    #print(node_id, end="  ")
    if node_id == (-1, 0):
        node = JointLimitAvoidance(
            name="jl",
            parent=None,
            calc_mappings=mappings.Identity(),
            gamma_p = 0.1,
            gamma_d = 0.05,
            lam = 1,
            sigma = 0.1,
            q_max = baxter.Common.q_max,
            q_min = baxter.Common.q_min,
            q_neutral = baxter.Common.q_neutral,
            parent_dim=7
        )
    else:
        temp_map = baxter.CPoint(*node_id)
        node = Node(
            name = 'x_' + str(node_id[0]) + '_' + str(node_id[1]),
            dim = 3,
            parent = None,
            mappings = temp_map
        )
        if node_id == (7, 0):
            ### 目標 ###
            g_dot = np.zeros_like(g)
            attracter = GoalAttractor(
                name="ee-attractor", parent=node, dim=3,
                calc_mappings=mappings.Translation(g, g_dot),
                max_speed = 5.0,
                gain = 10.0,
                f_alpha = 0.15,
                sigma_alpha = 1.0,
                sigma_gamma = 1.0,
                wu = 10.0,
                wl = 0.1,
                alpha = 0.15,
                epsilon = 0.5,
            )
            node.add_child(attracter)

        ### 障害物 ###
        for i, o in enumerate(o_s):
            obs_node = ObstacleAvoidance(
                name="obs_" + str(i) + ", at " + node.name,
                parent = node,
                calc_mappings = mappings.Distance(o, np.zeros_like(o)),
                scale_rep = 0.2,
                scale_damp = 1,
                gain = 50,
                sigma = 1,
                rw = 0.2
            )
            node.add_child(obs_node)

    node.isMulti = True
    f, M = node.solve(q, q_dot)
    #print(node.name)
    #print("  f = ", f.T)
    #print("  M = ", np.linalg.eigvals(M))
    return f, M



def solve(q, q_dot, g, o_s):
    #core=1
    core = cpu_count()-1
    with Pool(core) as p:
        ### プロセス毎にサブツリーを再構成して計算 ###
        node_ids = []
        node_ids.append((-1, 0))
        for i, Rs in enumerate(baxter.Common.R_BARS_ALL):
            node_ids += [(i, j) for j in range(len(Rs))]
        result = p.starmap(
            func = multi_solve2,
            iterable = ((node_id, q, q_dot, g, o_s) for node_id in node_ids)
        )
    
    
    f = np.zeros_like(result[0][0])
    M = np.zeros_like(result[0][1])
    for r in result:
        f += r[0]
        M += r[1]
    
    #print(self.f)
    q_ddot = np.linalg.pinv(M) @ f
    
    return q_ddot