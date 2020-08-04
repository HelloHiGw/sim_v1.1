import numpy as np

import time
import timeit

import cProfile
from line_profiler import LineProfiler
import sys

from SimInterface import SimInterface


def test():
    start = timeit.default_timer()
    sim = SimInterface(time=5, render=0, dev=0)
    input_orders = np.ones((10, 6), dtype=np.int8)
    sim.step(input_orders)
    sim.reset()
    end = timeit.default_timer()
    print("Elapsed (with compilation) init() + step() + reset() = %s" % (end - start))

    start = timeit.default_timer()
    for i in range(1):
        for j in range(100):
            sim.step(input_orders)
        sim.reset()
    end = timeit.default_timer()
    print("Elapsed (after compilation) step() x 100 + reset() x 10 = %s" % (end - start))

if __name__ == "__main__":
    # cProfile.run('test()',sort='tottime') # 测试程序所用时间
    # lp = LineProfiler(test) # 测试每行代码所用时间
    # lp.enable()
    # test()
    # lp.disable() 
    # lp.print_stats(sys.stdout)
    test()
    