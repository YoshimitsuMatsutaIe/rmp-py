"""三角形を維持"""
ROOBOT_R = 0.05
FORMATION_PRESERVATION_R = 0.2
PAIR_AVOIDANCE_R = 0.08
OBS_AVOIDANCE_R = 0.1

sim_param = {
    "trial" : 1,  #実験回数
    #"robot_model" : "car",
    "robot_model" : "turtlebot",
    "time_span" : 300,
    "time_interval" : 0.05,
    # "N" : 4,
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
    
    # "N" : 4,
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
    
    "N" : 4,
    "pair" : [[], [], [], []], #ペアなし
    
    "robot_r" : ROOBOT_R,  #ロボットの半径
    "robot_cpoints_num" : 4,
    "initial_condition" : {
        "type" : "random",
        "value" : {
            "x_max" : -0.5,
            "x_min" : 0.5,
            "y_max" : -0.5,
            "y_min" : 0.5,
        },
        "velocity" : {
            "type" : "fixed",
            "value" : "zero",
        }
    },
    #"goal_s" : [[0.5, 0.5], [], [], [],], #ゴールのみ
    
    "goal_s" : [[-0.15, -0.15], [-0.15, 0.15], [0.15, -0.15], [0.15, 0.15]],
    
    "obstacle_s" : [],
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
                "k" : 15,
                "alpha_psi" : 1,
                "k_d" : 50
            }
        }
    }
}

if __name__ == "__main__":
    pass