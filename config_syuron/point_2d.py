import math
from math import pi

COLLISION_R = 0.1
ROBOT_R = 0.05
FORMATION_PRESERVATION_R = 0.2
PAIR_AVOIDANCE_R = ROBOT_R*2
OBS_AVOIDANCE_R = ROBOT_R + COLLISION_R


ROBOT_NUM = 5

TASK_DIM = 2
X_MAX = 0.5
X_MIN = -0.5
Y_MAX = 0.5
Y_MIN = -0.5

sim_param = {
    "trial" : 1,  #実験回数
    "time_span" : 60,
    "time_interval" : 0.01,
    "task_dim" : TASK_DIM,
    "robot_num" : ROBOT_NUM,
    "collision_r" : COLLISION_R,
    "robot_r" : ROBOT_R,
    # "pair" : [
    #     [1, 4],
    #     [0, 2],
    #     [1, 3],
    #     [2, 4],
    #     [0, 3]
    # ],
    "pair" : [
        [1, 4],
        [0, 2, 4],
        [1, 3],
        [2, 4],
        [0, 3]
    ],
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
                "x_max" : X_MAX,
                "x_min" : X_MIN,
                "y_max" : Y_MAX,
                "y_min" : Y_MIN,
            }
        },
        "velocity" : {
            "type" : "zero"
        }
    },
    # "goal" : {
    #     "type" : "random",
    #     "value" : {
    #         "n" : ROBOT_NUM,
    #         "x_max" : X_MAX,
    #         "x_min" : X_MIN,
    #         "y_max" : Y_MAX,
    #         "y_min" : Y_MIN
    #     }
    # },
    "goal" : {
        "type" : "fixed",
        "value" : [
            [0, 0], [], [], [], []
        ]
        # "value" : [
        #     [], [], [], [], []
        # ]
    },
    # "obstacle" : {
    #     "type" : "random",
    #     "value" : {
    #         "n" : 5,
    #         "x_max" : X_MAX,
    #         "x_min" : X_MIN,
    #         "y_max" : Y_MAX,
    #         "y_min" : Y_MIN
    #     }
    # },
    "obstacle" : {
        "type" : "fixed",
        "value" : []
    },
    "controller" : {
        "rmp" : {
            "formation_preservation" : {
                "d" : FORMATION_PRESERVATION_R,
                "c" : 1,
                "alpha" : 5,
                "eta" : 100,
            },
            "pair_avoidance" : {
                "Ds" : PAIR_AVOIDANCE_R,
                "alpha" : 0.00001,
                "eta" : 0.2,
                "epsilon" : 0.00001,
            },
            "obstacle_avoidance" : {
                "Ds" : OBS_AVOIDANCE_R,
                "alpha" : 0.00001,
                "eta" : 0.2,
                "epsilon" : 0.00001,
            },
            "goal_attractor" : {
                "wu" : 2,
                "wl" : 0.2,
                "gain" : 50,
                "sigma" : 1,
                "alpha" : 1,
                "tol" : 0.001,
                "eta" : 50,
            }
        },
        "fabric" : {
            "formation_preservation" : {
                "d" : FORMATION_PRESERVATION_R,
                "m_u" : 2,
                "m_l" : 0.1,
                "alpha_m" : 0.75,
                "k" : 5,
                "alpha_psi" : 1,
                "k_d" : 100,
            },
            "angle_preservation" : {
                "m_u" : 2,
                "m_l" : 0.1,
                "alpha_m" : 0.75,
                "k" : 0.5,
                "alpha_psi" : 1,
                "k_d" : 0,
            },
            # pair_avoidance :
            #   r : *collision_pair_r
            #   k_b : 20
            #   alpha_b : 0.75
            # obstacle_avoidance:
            #   r : *collision_obs_r
            #   k_b : 29
            #   alpha_b : 1
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
                "k" : 20,
                "alpha_psi" : 1,
                "k_d" : 100,
                "dim" : TASK_DIM
            }
        }
    }
}