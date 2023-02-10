import math
from math import pi, cos, sin, sqrt
import numpy as np


COLLISION_R = 0.1
ROBOT_R = 0.05
FORMATION_PRESERVATION_R = 0.1
PAIR_AVOIDANCE_R = ROBOT_R*2
OBS_AVOIDANCE_R = ROBOT_R + COLLISION_R


ROBOT_NUM = 5

TASK_DIM = 2
X_MAX = 0.5
X_MIN = -0.5
Y_MAX = 0.5*2
Y_MIN = -0.5*2

def rotate(theta):
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])


# 五角形の計算
def pentagon():
    r = FORMATION_PRESERVATION_R/2/cos(54/180*pi)*1.2
    xs = []
    for i in [0, 1, 2, 3, 4]:
        x = [
            r*cos(2*pi/5*i + pi/2),# + np.random.rand()*0.5,
            r*sin(2*pi/5*i + pi/2),# + np.random.rand()*0.5
        ]
        xs.append(x)
    
    return xs

PENTA_R = FORMATION_PRESERVATION_R*cos(36/180*pi)*2

sim_param = {
    "trial" : 5,  #実験回数
    "time_span" : 30,
    "time_interval" : 0.01,
    "task_dim" : TASK_DIM,
    "robot_num" : ROBOT_NUM,
    "collision_r" : COLLISION_R,
    "robot_r" : ROBOT_R,
    "formation_preservation_r" : FORMATION_PRESERVATION_R,
    "penta_r" : PENTA_R,
    "space_limit" : {
        "x_max" : X_MAX,
        "x_min" : X_MIN,
        "y_max" : Y_MAX,
        "y_min" : Y_MIN,
    },
    "pair" : [
        [
            [1, FORMATION_PRESERVATION_R],
            [4, FORMATION_PRESERVATION_R]
        ],
        [
            [0, FORMATION_PRESERVATION_R],
            [2, FORMATION_PRESERVATION_R],
            [4, PENTA_R],
            [3, PENTA_R]
        ],
        [
            [1, FORMATION_PRESERVATION_R],
            [3, FORMATION_PRESERVATION_R]
        ],
        [
            [2, FORMATION_PRESERVATION_R],
            [4, FORMATION_PRESERVATION_R],
            [1, PENTA_R]
        ],
        [
            [0, FORMATION_PRESERVATION_R],
            [3, FORMATION_PRESERVATION_R],
            [1, PENTA_R]
        ]
    ], #五角形
    # "pair" : [
    #     [1, 4],
    #     [0, 2, 4],
    #     [1, 3],
    #     [2, 4],
    #     [0, 3]
    # ],
    # "pair" : [
    #     [1, 2],
    #     [0, 2, 3],
    #     [0, 1, 4],
    #     [1],
    #     [2]
    # ],  #鶴翼の陣

    #"pair" : [[] for _ in range(ROBOT_NUM)],
    
    # "angle_pair" : [
    #     [
    #         [1, 4, 108/180*pi]
    #     ],
    #     [
    #         [0, 2, 108/180*pi]
    #     ],
    #     [
    #         [1, 3, 108/180*pi]
    #     ],
    #     [
    #         [2, 4, 108/180*pi]
    #     ],
    #     [
    #         [0, 3, 108/180*pi]
    #     ]
    # ],  #正五角形
    
    # "angle_pair" : [
    #     [
    #         [1, 3, 1/2*pi]
    #     ],
    #     [
    #         [0, 2, 1/2*pi]
    #     ],
    #     [
    #         [1, 3, 1/2*pi]
    #     ],
    #     [
    #         [2, 0, 1/2*pi]
    #     ],
    # ],  #正五角形
    
    "angle_pair" : [[] for _ in range(ROBOT_NUM)],
    
    "initial_condition" : {
        "position" : {
            "type" : "random",
            "value" : {
                "x_max" : X_MAX-2*ROBOT_R,
                "x_min" : X_MIN+2*ROBOT_R,
                "y_max" : Y_MIN+6*ROBOT_R,
                "y_min" : Y_MIN+2*ROBOT_R,
            }
            # "type" : "fixed",
            # "value" : pentagon()
        },
        "velocity" : {
            "type" : "zero"
        }
    },
    "goal" : {
        "type" : "random",
        "value" : {
            "point" : [True, None, None, None, None],
            "x_max" : X_MAX-2*ROBOT_R,
            "x_min" : X_MIN+2*ROBOT_R,
            "y_max" : Y_MAX-2*ROBOT_R,
            "y_min" : Y_MAX-6*ROBOT_R
        }
    },
    # "goal" : {
    #     "type" : "fixed",
    #     "value" : [
    #         [0.6, 0.6], [], [], [], []
    #     ]
    #     # "value" : [
    #     #     [], [], [], [], []
    #     # ]
    # },
    "obstacle" : {
        "type" : "random",
        "value" : {
            "n" : 6,
            "x_max" : X_MAX, "x_min" : X_MIN,
            "y_max" : Y_MAX/2, "y_min" : Y_MIN/2
            # "x_max" : 0.8, "x_min" : 0,
            # "y_max" : 0.8, "y_min" : 0
        }
    },
    # "obstacle" : {
    #     "type" : "fixed",
    #     "value" : []
    # },
    "controller" : {
        "rmp" : {
            "formation_preservation" : {
                #"d" : FORMATION_PRESERVATION_R,
                "c" : 1,
                "alpha" : 100,
                "eta" : 100,
            },
            "pair_avoidance" : {
                "Ds" : PAIR_AVOIDANCE_R,
                "alpha" : 0.0000001,
                "eta" : 0.2,
                "epsilon" : 0.00001,
            },
            "obstacle_avoidance" : {
                "Ds" : OBS_AVOIDANCE_R,
                "alpha" : 0.0000001,
                "eta" : 0.2,
                "epsilon" : 0.00001,
            },
            "goal_attractor" : {
                "wu" : 2,
                "wl" : 0.2,
                "gain" : 200,
                "sigma" : 1,
                "alpha" : 1,
                "tol" : 0.001,
                "eta" : 100,
            }
        },
        "fabric" : {
            "formation_preservation" : {
                #"d" : FORMATION_PRESERVATION_R,
                "m_u" : 10,
                "m_l" : 0.1,
                "alpha_m" : 0.75,
                "k" : 20,
                "alpha_psi" : 1,
                "k_d" : 100,
            },
            # "angle_preservation" : {
            #     "m_u" : 2,
            #     "m_l" : 0.1,
            #     "alpha_m" : 0.75,
            #     "k" : 0.5,
            #     "alpha_psi" : 1,
            #     "k_d" : 100,
            # },
            # pair_avoidance :
            #   r : *collision_pair_r
            #   k_b : 20
            #   alpha_b : 0.75
            # "obstacle_avoidance" : {
            #     "r" : OBS_AVOIDANCE_R*1.5,
            #     "k_b" : 20,
            #     "alpha_b" : 1
            # },
            "pair_avoidance" : {
                "r" : PAIR_AVOIDANCE_R,
                "ag" : 100,
                "ap" : 100,
                "k" : 20,
            },
            "obstacle_avoidance" : {
                "r" : OBS_AVOIDANCE_R,
                "ag" : 100,
                "ap" : 100,
                "k" : 20,
            },
            "goal_attractor" : {
                "m_u" : 2,
                "m_l" : 0.2,
                "alpha_m" : 1,
                "k" : 200,
                "alpha_psi" : 1,
                "k_d" : 100,
                "dim" : TASK_DIM
            },
            # "space_limit_avoidance" : {
            #     "r" : 0.1,
            #     "sigma" : 10,
            #     "a" : 10,
            #     "k" : 0.001,
            #     "x_max" : X_MAX,
            #     "x_min" : X_MIN,
            #     "y_max" : Y_MAX,
            #     "y_min" : Y_MIN
            # }
            "space_limit_avoidance" : {
                "gamma_p" : 0.8,
                "gamma_d" : 5,
                "lam" : 10,
                "sigma" : 2,
                "x_max" : X_MAX - ROBOT_R,
                "x_min" : X_MIN + ROBOT_R,
                "x_0" : 0,
                "y_max" : Y_MAX +  ROBOT_R,
                "y_min" : Y_MIN + ROBOT_R,
                "y_0" : 0
            }
        }
    }
}