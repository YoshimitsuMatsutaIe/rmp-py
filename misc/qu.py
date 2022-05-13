# from multiprocessing import Process, Queue
# import os
# import time
# import random


# # Queueにデータを書き込む
# def write(q):
#     print('Process to write: {}'.format(os.getpid()))
#     for value in ['A', 'B', 'C']:
#         print('Put {} to queue...'.format(value))
#         q.put(value)
#         time.sleep(random.random())


# # Queueからデータを読み取り
# def read(q):
#     print('Process to read: {}'.format(os.getpid()))
#     while True:
#         value = q.get(True)
#         print('Get {} from queue.'.format(value))


# # 親プロセスがQueueを作って、子プロセスに渡す
# q = Queue()
# pw = Process(target=write, args=(q,))
# pr = Process(target=read, args=(q,))
# # pwを起動し、書き込み開始
# pw.start()
# # prを起動し、読み取り開始
# pr.start()
# # pwが終了するのを待つ
# pw.join()
# # prは無限ループなので、強制終了
# pr.terminate()



import concurrent.futures
import time

numbers = [i for i in range(100)]
result = [0 for i in range(100)]
print(result)

def Calc(number):
    #何か重い処理
    print(number)
    result[number] = number
    time.sleep(1)


start = time.time()

#multi process
pool = concurrent.futures.ProcessPoolExecutor(max_workers=11)
result = list(pool.map(Calc,numbers))

# #single process
# result=list(map(Calc,numbers))

print(result)

end = time.time()
print('%.3f' %(end-start))