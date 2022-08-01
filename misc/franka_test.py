import numpy as np
import matplotlib.pyplot as plt

from typing import Union


import sys
sys.path.append('.')


from franka_emika.htm import *

from franka_emika.franka_emika import *



q = np.array([[0, 30, 0, 30, 0, 45, 0]]).T * np.pi/180
dq = np.zeros_like(q)



max_x = 0.6
min_x = -0.6
max_y = 0.6
min_y = -0.6
max_z = 1
min_z = 0

mid_x = (max_x + min_x) * 0.5
mid_y = (max_y + min_y) * 0.5
max_range = max(max_x-min_x, max_y-min_y, max_z-min_z) * 0.5
mid_z = (max_z + min_z) * 0.5



fig = plt.figure()
ax = fig.add_subplot(projection='3d')




body = np.concatenate([
    np.zeros((3, 1)),
    o_0(q),
    o_1(q),
    o_2(q),
    o_3(q),
    o_4(q),
    o_5(q),
    o_6(q),
    o_ee(q),
], axis=1)


print(body)

ax.plot(body[0,:], body[1,:], body[2,:], marker='o', label="body")



for n, r_bars in enumerate(Common.R_BARS_ALL):
    cs = []
    for r_bar in r_bars:
        cs.append((Common.HTM[n](q) @ r_bar)[:3, :])
    cs = np.concatenate(cs,axis=1)
    ax.scatter(cs[0,:], cs[1,:], cs[2,:], label=str(n),)


ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_box_aspect((1,1,1))
ax.legend()
plt.show()