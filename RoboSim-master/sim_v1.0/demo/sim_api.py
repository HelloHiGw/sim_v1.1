import numpy as np

import threading as TD
import timeit
from queue import Queue

# 测试代码运行时间模块
import cProfile
import sys
import time

# Numba模块
from numba import jit,njit

from SimInterface import SimInterface

import guide
import car

def RunGame(cars_info, guide_info, endFlag):
    """
    仿真环境运行(get指示信息,put车辆信息)
    :return: None
    """

    game = SimInterface(agent_num=10, render=True)
    game.reset()
    # only when render = True
    game.play(cars_info, guide_info, endFlag)


def MakeStrategy(cars_info, guide_info, endFlag):
    """
    决策函数(get车辆信息,put指示信息)
    :return: None
    """

    # fake data //begin
    path0 = [[88, 38], [113, 38], [138, 38], [163, 38], [188, 38], [213, 38], [238, 38], [263, 38], [288, 38], [313, 38]]
    path1 = [[88, 188], [113, 188], [138, 213], [163, 238], [188, 238], [213, 238], [238, 238], [263, 238], [288, 213], [313, 188]]
    path2 = [[88, 313], [113, 313], [138, 313], [163, 313], [188, 313], [213, 313], [238, 313], [263, 313], [288, 313], [313, 313]]
    path3 = [[88, 438], [113, 413], [138, 388], [163, 363], [188, 363], [213, 363], [238, 363], [263, 388], [288, 413], [313, 438]]
    path4 = [[88, 563], [113, 563], [138, 563], [163, 563], [188, 563], [213, 563], [238, 563], [263, 563], [288, 563], [313, 563], [338, 563], [363, 563], [388, 563], [413, 563], [438, 563], [463, 563], [488, 563], [513, 563]]

    guide0 = guide.Guide(path0)
    guide1 = guide.Guide(path1)
    guide2 = guide.Guide(path2)
    guide3 = guide.Guide(path3)
    guide4 = guide.Guide(path4)

    allguide = [guide0, guide1, guide2, guide3, guide4] # list of class [Guide], see guide.py
    # fake data //end

    guide_info.put(allguide)

    t0 = time.time()
    while True:
        # 决策部分 //begin
        # ...
        # and delete fake data
        # 决策部分 //end
        # guide_info.put(allguide)

        if not endFlag.empty(): # 仿真环境结束后退出决策部分
            if endFlag.get() == 1:
                break

        if cars_info.empty():
            time.sleep(0.1) # 防止该线程一直占用
            continue
        cars = cars_info.get() # list of class [Car], see car.py
        
    t1 = time.time()
    print('Game runing time:', t1-t0, 's')
        
    
def multithreding():
    """
    双线程运行仿真和决策系统
    :return: None
    """

    cars_info = Queue(maxsize=1)
    guide_info = Queue(maxsize=1)
    endFlag = Queue(maxsize=1)
    t_game = TD.Thread(target=RunGame, args=(cars_info,guide_info,endFlag))
    t_strategy = TD.Thread(target=MakeStrategy, args=(cars_info,guide_info,endFlag))

    t_game.start()
    t_strategy.start()


def test5v5():
    env = SimInterface(agent_num=10, time=100, render=True)
    env.reset()

    fake_orders = np.array([[1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1], 
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1],
                            [1,1,1,1,1,0,0,1]],
                            dtype='int8')
    time.sleep(5)
    for i_step in range(100):
            done = env.step(fake_orders)
            print('step-', i_step, ':', done)


if __name__ == "__main__":
    #cProfile.run('test5v5()',sort='tottime')
    test5v5()
