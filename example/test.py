"""configよんでシミュレーション実行"""


import xmltodict

import sys
sys.path.append('.')
import simulator



with open("./config/franka_ex.xml") as f:
    param = f.read()
    param = xmltodict.parse(param)
    param = param["simulation_param"]


simulator.main(param)