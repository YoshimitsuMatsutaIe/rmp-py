"""configよんでシミュレーション実行"""

import sys
sys.path.append('.')
from simulator import Simulator


s = Simulator()

#simulator.main("./config/franka.json")

s.main("./config/franka_sphere.json")

#simulator.main("./config/baxter.json")


#simulator.main("./config/franka_box.json")

#s.main("./config/franka_cubbie.json")


#simulator.main("./config/test_sice.json")


#simulator.main("./../rmp-cpp/config/sice.json")



# cppのやつ
#simulator.main("../rmp-cpp/config/sice.json")
#s.main("../rmp-cpp/config/franka_sphere.json")