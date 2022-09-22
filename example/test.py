"""configよんでシミュレーション実行"""

import sys
sys.path.append('.')
from simulator import Simulator


s = Simulator()

#simulator.main("./config/franka.json")

#s.main("./config/franka_sphere.json")

#simulator.main("./config/baxter.json")


s.main("./config/franka_sphere.json")

#s.main("./config/test_sice.json", method="single")



#simulator.main("./../rmp-cpp/config/sice.json")

#s.main("./config/particle_test.json")



# cppのやつ
#simulator.main("../rmp-cpp/config/sice.json")
#s.main("../rmp-cpp/config/franka_sphere.json")