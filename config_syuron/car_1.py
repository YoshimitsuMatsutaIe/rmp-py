"""三角形を維持"""
import math
from math import pi
ROOBOT_R = 0.05
FORMATION_PRESERVATION_R = 0.2
PAIR_AVOIDANCE_R = 0.1
OBS_AVOIDANCE_R = ROOBOT_R*2

ROBOT_NUM = 3

X_MAX = 0.5
X_MIN = -0.5
Y_MAX = 0.5
Y_MIN = -0.5

sim_param = {
    "trial" : 1,  #実験回数
    "robot_model" : "turtlebot",
    #"robot_model" : "car",
    "time_span" : 120,
    "time_interval" : 0.01,
    "N" : ROBOT_NUM,
    
    "pair" : [[] for _ in range(ROBOT_NUM)],
    
    # "pair" : [
    #     [
    #         [2, [1, 0]],
    #     ],# 0番目．リーダー
    #     [
    #         [0, [0, 2]],
    #         #[2, [2, 0]]
    #     ],# 1番目．左
    #     [
    #         [0, [1, 2]],
    #         #[1, [1, 3]]
    #     ],# 2番目．右
    #     [
    #         [0, [2, 2]],
    #     ]
    # ],  # ケツに追従
    
    # "pair" : [
    #     [
    #         [1, [1, 0]],
    #         [3, [3, 0]]
    #     ],# 0番目．リーダー
    #     [
    #         [0, [0, 1]],
    #         [2, [2, 1]]
    #     ],# 1番目．左
    #     [
    #         [1, [1, 2]],
    #         [3, [3, 2]]
    #     ],# 2番目．右
    #     [
    #         [0, [0, 3]],
    #         [2, [2, 3]]
    #     ]
    # ],  # 四角形
    
    # "N" : 4,
    # "pair" : [[], [], [], []], #ペアなし
    
    # "pair" : [
    #     [
    #         [2, [1, 0]],
    #         [2, [2, 0]]
    #     ],# 0番目．リーダー
    #     [
    #         [0, [0, 2]],
    #         [3, [2, 1]]
    #     ],# 1番目．左
    #     [
    #         [0, [0, 2]],
    #         [1, [1, 3]]
    #     ],# 2番目．右
    # ], #三角形
    
    #"N" : 2,
    #"pair" : [[], []],
    
    # "N" : 1,
    # "pair" : [[]],
    "initial_condition" : {
        "type" : "fixed",
        "value" : [0.5, 0.0, pi/2, 0, 0, 0],
    },
    # "goal" : {
    #     "type" : "fixed",
    #     "value" : [
    #         [0.5, 0.0]
    #     ]
    # },
    # "obs" : {
    #     "type" : "fixed",
    #     "value" : [
    #         [0.2, 0.3],
    #     ]
    # },
    
    "obs" : {
        "type" : "none"
    },
    
    
    "robot_r" : ROOBOT_R,  #ロボットの半径
    "robot_cpoints_num" : 4,
    
    # "initial_condition" : {
    #     "type" : "random",
    #     "value" : {
    #         "x_max" : X_MAX,
    #         "x_min" : X_MIN,
    #         "y_max" : Y_MAX,
    #         "y_min" : Y_MIN,
    #     },
    #     "velocity" : {
    #         "type" : "fixed",
    #         "value" : "zero",
    #     }
    # },

    # "goal" : {
    #     "type" : "random",
    #     "value" : {
    #         "x_max" : X_MAX,
    #         "x_min" : X_MIN,
    #         "y_max" : Y_MAX,
    #         "y_min" : Y_MIN,
    #     }
    # },
    
    # "goal" : {
    #     "type" : "fixed",
    #     "value" : [[] for _ in range(ROBOT_NUM)],
    # },
    
    "goal" : {
        "type" : "fixed",
        "value" : [
            [0, 0],
        ]
    },
    
    # "goal" : {
    #     "type" : "fixed",
    #     "value" : [
    #         [-0.15, -0.15],
    #         [-0.15, 0.15],
    #         [0.15, -0.15],
    #         [0.15, 0.15]
    #     ],
    # },
    
    # "obs" : {
    #     "type" : "random",
    #     "value" : {
    #         "n" : 4,
    #         "x_max" : X_MAX,
    #         "x_min" : X_MIN,
    #         "y_max" : Y_MAX,
    #         "y_min" : Y_MIN,
    #         }
    # },
    
    "controller" :{
        "rmp" : {
            "formation_preservation" : {
                "d" : FORMATION_PRESERVATION_R,
                "c" : 1,
                "alpha" : 50,
                "eta" : 50,
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
                "epsilon" : 0.00001
            },
            "goal_attractor" : {
                "wu" : 10,
                "wl" : 0.1,
                "gain" : 15,
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
                "k" : 0.5,
                "alpha_psi" : 1,
                "k_d" : 50,
            },
            "pair_avoidance" : {
                "r" : PAIR_AVOIDANCE_R,
                "k_b" : 1,
                "alpha_b" : 0.75,
            },
            "obstacle_avoidance" : {
                "r" : OBS_AVOIDANCE_R,
                "k_b" : 1,
                "alpha_b" : 0.75,
            },
            "goal_attractor" :{
                "m_u" : 2,
                "m_l" : 0.2,
                "alpha_m" : 0.75,
                "k" : 100,
                "alpha_psi" : 1,
                "k_d" : 50
            }
        }
    }
}

if __name__ == "__main__":
    pass