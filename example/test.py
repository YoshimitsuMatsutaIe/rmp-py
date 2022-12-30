"""configよんでシミュレーション実行"""

import sys
sys.path.append('.')
from rmp_simulator import Simulator

s = Simulator()


#simulator.main("./config/franka.json")

#s.run_rmp_sim("./config/franka_sphere.json")

#s.environment_visualization("./config/franka_wall.json")
#s.run_rmp_sim("./config/franka_wall.json")
#s.run_rmp_sim("./config/baxter.json")

#s.run_rmp_sim("./config/franka_sphere.json")

s.run_rmp_sim("./config/test_sice.json", method="single")
#s.run_rmp_sim("./config/sice_non_obs.json")


#simulator.main("./../rmp-cpp/config/sice.json")

#s.run_rmp_sim("./config/particle_test.json")

#s.run_rmp_sim("./config/franka_cubbie.json")

# cppのやつ
#simulator.main("../rmp-cpp/config/sice.json")
#s.run_rmp_sim("../rmp-cpp/config/franka_sphere.json")





#s.environment_visualization("./config/yamanaka_test.json")
#s.run_rmp_sim("./config/yamanaka_test.json")



#s.environment_visualization("./config/sice_ex_test.json")
#s.run_rmp_sim("./config/sice_ex_test.json")
#s.run_rm/home/matsuta/src/rmp-py/robot_sice_exp_sim("./config/particle_have_initial.json")



#s.run_planning_sim("./config/sice_ex_test.json")
#s.run_planning_sim("./config/franka_sphere.json")




#s.run_rmp_sim("./config/sice_ex_e.json")

#s.run_rmp_sim("./config/fe_sphere2.json")






rmp_param = {
    "goal_attractor" : {
        "max_speed" : 10.0,
        "gain" : 20.0,
        "sigma_alpha" : 0.5,
        "sigma_gamma" : 0.5,
        "wu" : 10.0,
        "wl" : 0.1,
        "alpha" : 0.1,
        "epsilon" : 0.5
    },
    "obstacle_avoidance" : {
        "scale_rep" : 0.1,
        "scale_damp" : 1.0,
        "gain" : 6.0,
        "sigma" : 1.0,
        "rw" : 0.1
    },
    "joint_limit_avoidance" : {
        "gamma_p" : 0.05,
        "gamma_d" : 0.05,
        "lam" : 1.0,
        "sigma" : 0.1
    }
}

param = {
    "json_name" : "temp.json",
    "//" : "franka emika box test",
    "robot_name" : "franka_emika",
    "time_span" : 4.0,
    "time_interval" : 0.01,
    "rmp_param" : rmp_param,
    "env_param" : {
        "goal" : {
            "type" : "static",
            "param" : {
                "x" : 0.5,
                "y" : -0.01,
                "z" : 0.7
            }
        },
        "obstacle" : [
            {
                "type" : "field",
                "param" : {
                    "lx" : 1.0,
                    "ly" : 1.0,
                    "x" : -0.2,
                    "y" : 0.0,
                    "z" : 0.6,
                    "n" : 300,
                    "alpha" : 0.0, "beta" : 90.0, "gamma" : 0.0
                }
            },
            {
                "type" : "box",
                "param" : {
                    "lx" : 0.3,
                    "ly" : 1.0,
                    "lz" : 0.4,
                    "x" : 0.4,
                    "y" : 0.0,
                    "z" : 0.3,
                    "n" : 1000,
                    "alpha" : 0.0, "beta" : 0.0, "gamma" : 0.0
                }
            },
            {
                "type" : "box",
                "param" : {
                    "lx" : 0.2,
                    "ly" : 1.0,
                    "lz" : 0.335,
                    "x" : 0.4,
                    "y" : 0.0,
                    "z" : 1.0,
                    "n" : 300,
                    "alpha" : 0.0, "beta" : 0.0, "gamma" : 0.0
                }
            }
        ]
    }
}




# s.main(param_dict = param)

