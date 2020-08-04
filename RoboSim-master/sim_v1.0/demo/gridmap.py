import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import point

class GridMap:
    def __init__(self, L=32, W=24):
        self.L = L # 地图长度
        self.W = W # 地图宽度
        self.ratio = 800//L # 比例尺,默认地图尺寸800*600
        self.carSize = 60//self.ratio+1 # 车辆尺寸
        self.obstacle_point = [] # 障碍物位置
        self.GenerateObstacle() # 生成障碍物


    def GenerateObstacle(self):
        """
        在地图中生成障碍物
        :return 
        """

        ratio = self.ratio
        # 中间障碍O1
        for i in range(350//ratio+1, 450//ratio+1):
            for j in range(250//ratio+1, 350//ratio+1):
                self.obstacle_point.append(point.Point(i, j))
                self.obstacle_point.append(point.Point(i, j))
        
        # 左侧障碍O2,O3
        for i in range(175//ratio+1, 225//ratio+1):
            for j in range(75//ratio+1, 200//ratio+1):
                self.obstacle_point.append(point.Point(i, j))
                self.obstacle_point.append(point.Point(i, j))
            for j in range(400//ratio+1, 525//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j))

        for i in range(225//ratio+1, 300//ratio+1):
            for j in range(75//ratio+1, 125//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j))
            for j in range(475//ratio+1, 525//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j)) 
        
        # 右侧障碍O4,O5
        for i in range(575//ratio+1, 625//ratio+1):
            for j in range(75//ratio+1, 200//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j))
            for j in range(400//ratio+1, 525//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j))

        for i in range(500//ratio+1, 575//ratio+1):
            for j in range(75//ratio+1, 125//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j))
            for j in range(475//ratio+1, 525//ratio+1):
                self.obstacle_point.append(point.Point(i, j)) 
                self.obstacle_point.append(point.Point(i, j))


    def IsObstacle(self, x, y):
        """
        是否是障碍
        :param x: 待检查点-x坐标
        :param y: 待检查点-y坐标
        :return 
        """

        for p in self.obstacle_point:
            if x == p.x and y == p.y:
                return True
        return False


    def IsCarHitObstacle(self, x, y):
        """
        车辆是否与障碍碰撞
        :param x: 待检查点-x坐标
        :param y: 待检查点-y坐标
        :return 
        """

        for p in self.obstacle_point:
            if abs(x-p.x) <= self.carSize//2 and abs(y-p.y) <= self.carSize//2:
                return True
        return False
    
if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    plt.ion() # 打开交互模式
    # 生成随机地图
    map = GridMap(32, 24)
    ax = plt.gca()
    ax.set_xlim([0, map.L])
    ax.set_ylim([0, map.W])
    
    plt.cla() # 清除原有图像
    for i in range(map.L): # 绘制地图、障碍、己方车辆和敌方车辆
        for j in range(map.W):
            if map.IsObstacle(i+1, j+1):
                rec = Rectangle((i, j), width=1, height=1, color='gray')
                ax.add_patch(rec)
            else:
                rec = Rectangle((i, j), width=1, height=1, edgecolor='gray', facecolor='w')
                ax.add_patch(rec)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
    plt.ioff() # 关闭交互模式
    plt.show()