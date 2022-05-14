from multiprocessing import Pool
import os
import time
import random
import numpy as np

# def long_time_task(name):
#     print('Run task {} ({})...'.format(name, os.getpid()))
#     start = time.time()
#     print(np.sum((np.random.random((name*100, name*100))**5)))
#     time.sleep(0.5)
#     end = time.time()
#     print('Task {} runs {} seconds.'.format(name, (end - start)))


# print('Parent process {}.'.format(os.getpid()))
# p = Pool(40)  # 同時に最大4個の子プロセス
# for i in range(40):
#     p.apply_async(long_time_task, args=(i,))
# # 非同期処理のため、親プロセスは子プロセスの処理を待たずに、
# # 次のprintをする
# print('Waiting for all subprocesses done...')
# p.close()
# p.join()
# print('All subprocesses done.')



### シンプル ###
def sample_func(initial_num, h):
    name = str(initial_num)
    for i in range(100):
        initial_num += 1
    hoge = np.array([[h, h]])
    return name, initial_num, hoge


initial_num_list = list(range(100))
h_list = [2*i for i in initial_num_list]

# for i in initial_num_list:
#     sample_func(initial_num=i)


with Pool() as p:
    whw = p.starmap(
        func=sample_func,
        iterable=[(n, h) for n, h in zip(initial_num_list, h_list)]
    )

print(whw)

print(whw[0][2] + whw[1][2])