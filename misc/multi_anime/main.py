from unittest import result
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import numpy as np
import time
from multiprocessing import Pool, cpu_count


# x = np.arange(0, 2*np.pi, 0.0001)
# y = np.sin(x)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlim(0, 2*np.pi)
# ax.set_ylim(-1.1, 1.1)
# ax.grid()
# ax.legend()
# ax.plot(x, y)
# t0 = time.time()
# artists = []
# for i in range(len(x)):


#     artist = ax.plot(x[i], y[i],"blue", marker="o")
#     artists.append(artist)

# print("art time = ", time.time() - t0)

# t1 = time.time()
# # 4. アニメーション化
# ani = anm.ArtistAnimation(fig, artists)

# print("ani time = ", time.time() - t1)

# plt.show()



x = np.arange(0, 2*np.pi, 1)
y = np.sin(x)

x = x.tolist()
y = y.tolist()


def callback(x, y, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.1, 1.1)
    ax.plot(x, y, label="line")
    ax.plot(x[i], y[i], "blue", marker="o")
    ax.grid()
    ax.legend()
    return [ax]


fig = plt.figure(111)
ax = fig.add_subplot()

t0 = time.time()
core = cpu_count()
#core = 1
with Pool(core) as p:
    result = p.starmap(
        func = callback,
        iterable = ((x, y, i) for i in range(len(x)))
    )



print("art time = ", time.time() - t0)

t1 = time.time()

fig = plt.figure()

ani = anm.ArtistAnimation(fig, result)

print("ani time = ", time.time() - t1)

ani.save('ani1.gif', writer='pillow')

plt.show()






# x = np.arange(0, 2*np.pi, 1)
# y = np.sin(x)

# x = x.tolist()
# y = y.tolist()


# def callback(x, y, i):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1.1, 1.1)
#     ax.plot(x, y, label="line")
#     ax.plot(x[i], y[i], "blue", marker="o")
#     ax.grid()
#     ax.legend()
#     fig.savefig("misc/multi_anime/fig/" + str(i) + ".png")



# t0 = time.time()
# core = cpu_count()
# #core = 1
# with Pool(core) as p:
#     result = p.starmap(
#         func = callback,
#         iterable = ((x, y, i) for i in range(len(x)))
#     )


# import glob
# from PIL import Image



# picList = glob.glob("misc/multi_anime/fig/*.png")
# print(picList)
# ims = []
# for i in range(len(picList)):
#     #tmp = Image.open(picList[i])
#     tmp = plt.imread(picList[i])
#     ims.append([plt.imshow(tmp)])




# print("art time = ", time.time() - t0)

# t1 = time.time()


# fig = plt.figure()

# ani = anm.ArtistAnimation(fig, ims)

# print("ani time = ", time.time() - t1)

# ani.save('ani1.gif', writer='pillow')

# #plt.show()



# x = np.arange(0, 2*np.pi, 0.1)
# y = np.sin(x)

# x = x.tolist()
# y = y.tolist()


# def callback(x, y, i):
#     i1, = plt.plot(x, y, label="line")
#     i2, = plt.plot(x[i], y[i], "blue", marker="o")
    
#     return [i1, i2]




# t0 = time.time()
# core = cpu_count()
# #core = 1
# with Pool(core) as p:
#     result = p.starmap(
#         func = callback,
#         iterable = ((x, y, i) for i in range(len(x)))
#     )



# print("art time = ", time.time() - t0)

# t1 = time.time()


# fig = plt.figure()

# ani = anm.ArtistAnimation(fig, result)

# print("ani time = ", time.time() - t1)

# ani.save('ani1.gif', writer='pillow')

# #plt.show()



