from operator import index
from typing import Dict
import robot_franka_emika.franka_emika as franka_emika
import robot_baxter.baxter as baxter
import robot_sice.sice as sice
import robot_sice_ex.sice_ex as sice_ex
import robot_particle.particle as particle
import robot_yamanaka.yamanaka as yamanaka

def get_robot_model(robot_name):
    if robot_name == "baxter":
        return baxter
    elif robot_name == "franka_emika":
        return franka_emika
    elif robot_name == "sice":
        return sice
    elif robot_name == "sice_ex":
        return sice_ex
    elif robot_name == "particle":
        return particle
    elif robot_name == "yamanaka":
        return yamanaka
    else:
        assert False


def get_cpoint_ids(rm, ex_param):
    """get control point index"""
    c_ = rm.CPoint(0, 0, **ex_param)
    node_ids = []
    for i, Rs in enumerate(c_.RS_ALL):
        node_ids += [(i, j) for j in range(len(Rs))]
    
    return node_ids





class KinematicsAll:
    def __init__(self, robot_name, ex_param={}):
        self.rm = get_robot_model(robot_name)
        node_ids = get_cpoint_ids(self.rm, ex_param)
        self.cpoint_maps = [
            self.rm.CPoint(*node_id, **ex_param) for node_id in node_ids
        ]
        c_ = self.rm.CPoint(0, 0, **ex_param)
        self.ee_map = self.rm.CPoint(
            *c_.ee_id, **ex_param
        )
    
    def calc_cpoint_position_all(self, q):
        os = []
        for m in self.cpoint_maps:
            os.append(m.phi(q))
        
        return os

    def calc_ee_position(self, q):
        return self.ee_map.phi(q)


if __name__ == "__main__":
    import numpy as np
    K = KinematicsAll("franka_emika")
    K = KinematicsAll("sice_ex", {"c_dim": 5, "total_length": 4.0})
