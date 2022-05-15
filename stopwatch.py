import time

def print_proc_time(func):
    """CPU時間計測デコレータ"""
    def print_proc_time_func(*args, **kwargs):
        t0 = time.process_time()
        return_val = func(*args, **kwargs)
        t1 = time.process_time()
        elapsed_time = t1 - t0
        print(func.__name__, elapsed_time)
        return return_val
    return print_proc_time_func


def print_pref_time(func):
    """実行時間計測デコレータ"""
    def print_proc_time_func(*args, **kwargs):
        t0 = time.perf_counter()
        return_val = func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        print(func.__name__, elapsed_time)
        return return_val
    return print_proc_time_func




if __name__ == "__main__":
    @print_pref_time
    def test(x):
        return [i for i in range(x)]
    
    
    x = 1000
    
    out1 = test(x)