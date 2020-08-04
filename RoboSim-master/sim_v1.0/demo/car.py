import numpy as np
from numba import types, typed
from numba.experimental import jitclass

# spec_Car = [
#     ('team', types.int32),
#     ('x', types.float32),
#     ('y', types.float32),
#     ('rotate', types.float32),
#     ('yaw', types.float32),
#     ('HP', types.int32),
#     ('bullet', types.int32),
# ]

# @jitclass(spec_Car)
# class Car(object):
#     def __init__(self, team, x, y, rotate, yaw, HP, bullet):
#         self.team   = team # 队伍
#         self.x      = x # x坐标
#         self.y      = y # y坐标
#         self.rotate = rotate # 底盘方向角
#         self.yaw    = yaw # 云台方向角
#         self.HP     = HP # 生命值
#         self.bullet = bullet # 子弹数   


class Car1(object):
    def __init__(self, team, x, y, rotate, yaw, HP, bullet):
        self.team   = team # 队伍
        self.x      = x # x坐标
        self.y      = y # y坐标
        self.rotate = rotate # 底盘方向角
        self.yaw    = yaw # 云台方向角
        self.HP     = HP # 生命值
        self.bullet = bullet # 子弹数 
