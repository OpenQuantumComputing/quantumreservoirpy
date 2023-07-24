from multiprocessing import Pool
import time
from functools import wraps



def timeit(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("@timefn: {} took {} seconds.".format(func.__name__, end_time - start_time))
        return result
    return measure_time


def isprime(num):
    if num % 2 == 0:
        return False
    for j in range(3, num//2):
        if num % j == 0:
            return False
    return True


# with Pool() as pool:

@timeit
def without(series):
    for elem in series:
        isprime(elem)

@timeit
def withpool(series):
    with Pool() as pool:
        n = pool.map(isprime, series)

    # for bool, val in zip(n, series):
    #     if bool:
    #         print(val)

def main():
    res = [i for i in range(100000)]
    without(res)
    withpool(res)


if __name__ == '__main__':
    main()