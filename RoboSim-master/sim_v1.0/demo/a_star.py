import sys
import time
import random

import numpy as np

import point
import gridmap
import convertc

class AStar:
    class Node: # AStar算法中的节点数据
        def __init__(self, point, endPoint, g=0):
            self.point = point # 自己的坐标
            self.father = None # 父节点
            self.g = g # 自己距离起点的距离
            x_dis = abs(endPoint.x - point.x) # x方向距离
            y_dis = abs(endPoint.y - point.y) # y方向距离
            self.h =  x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis) # 启发函数h，距终点的欧式距离


    def __init__(self, map, startPoint, endPoint):
        self.map = map # 地图
        self.carSize = map.carSize # 车辆尺寸
        self.startPoint = startPoint # 起始点坐标
        self.endPoint = endPoint # 终点坐标

        self.open_list = [] # 开列表
        self.close_list = [] # 闭列表
        self.res_path = [] # 结果路径


    def update(self, map, startPoint, endPoint, selfNum):
        self.map = map
        self.carSize = map.carSize
        self.startPoint = startPoint
        self.endPoint = endPoint     
        
        self.open_list = []
        self.close_list = []
        self.res_path = []


    def getMinNode(self):
        """
        获得openlist中F值最小的节点
        :return: Node
        """

        currentNode = self.open_list[0]
        for node in self.open_list:
            if node.g + node.h < currentNode.g + currentNode.h:
                currentNode = node
        return currentNode


    def GetPosWithG(self, path_list, g):
        """
        根据g值找出路径上最接近的点
        :param path_list: 待检查路径点列表
        :param g: (x,y)距离起点的距离
        :return: Bool
        """

        g_cur = 0 # 当前g值
        pathLen = len(path_list)

        if pathLen == 0 or pathLen == 1: # 路径列表为空或仅有单个点
            return None
        else: 
            for i in range(pathLen-1):
                p_now = path_list[i]
                p_next = path_list[i+1]
                g_cur = g_cur + ((p_next.x-p_now.x) ** 2 + (p_next.y-p_now.y) ** 2) ** 0.5 # 计算新的g值
                dg = g - g_cur
                if dg < 1.5:
                    return p_next # 找到相近点并返回
                else:
                    return None # 未找到相近点，不会发生路径冲突


    def IsValidPoint(self, curNode, x, y, step):
        """
        检查是否为可行点
        :param curNode: AStar算法计算过程中当前节点
        :param x: 待检查点-x坐标
        :param y: 待检查点-y坐标
        :param step: 当前节点至(x,y)的距离
        :return: Bool
        """

        # 是否越界
        if x < 1 or y < 1:
            return False
        if x > self.map.L or y > self.map.W:
            return False
        # 是否为障碍
        if self.map.IsCarHitObstacle(x, y):
            return False
        return True


    def pointInPointList(self, p, point_list):
        """
        检查该点是否在点列表中
        :param p: 待检查节点
        :param point_list: 待搜索点列表
        :return: Node
        """

        for node in point_list:
            if node.point == p:
                return node
        return None


    def pointInOpenList(self, p):
        """
        检查该点是否在开列表中
        :param p: 待检查节点
        :return: Node
        """

        return self.pointInPointList(p, self.open_list)


    def pointInCloseList(self, p):
        """
        检查该点是否在闭列表中
        :param p: 待检查节点
        :return: Node
        """

        return self.pointInPointList(p, self.close_list)


    def endPointInCloseList(self): # 终点是否在闭列表中
        """
        检查终点是否在闭列表中
        :return: Node
        """

        for node in self.close_list:
            if node.point == self.endPoint:
                return node
        return None


    def searchNear(self, minF, offsetX, offsetY, step):
        """
        搜索节点周围的点
        :param minF: F值最小的节点
        :param offsetX: X坐标偏移量
        :param offsetY: Y坐标偏移量
        :param step: 步长(1 or sqrt(2))
        :return:
        """

        # 可行性检测
        if not self.IsValidPoint(minF, minF.point.x + offsetX, minF.point.y + offsetY, step):
            return
        # 如果在close_list中，忽略该节点
        currentPoint = point.Point(minF.point.x + offsetX, minF.point.y + offsetY)
        if self.pointInCloseList(currentPoint):
            return
        # 如果不在open_list中，就把它加入open_list
        currentNode = self.pointInOpenList(currentPoint)
        if not currentNode:
            currentNode = AStar.Node(currentPoint, self.endPoint, g=minF.g+step)
            currentNode.father = minF
            self.open_list.append(currentNode)
            return
        # 如果在open_list中，判断minF到当前点的g是否更小
        if minF.g + step < currentNode.g:  # 如果更小，就重新计算g值，并且更改父节点
            currentNode.g = minF.g + step
            currentNode.father = minF


    def start(self):
        """
        开始寻路
        :return: None或Point列表（路径）
        """

        # 判断寻路终点是否可行
        if not self.IsValidPoint(None, self.endPoint.x, self.endPoint.y, 0):
            return []
 
        # 1.将起点放入开启列表
        startNode = AStar.Node(self.startPoint, self.endPoint)
        self.open_list.append(startNode)
        # 2.主循环逻辑
        while True:
            # 找到F值最小的点
            minF = self.getMinNode()
            # 把这个点加入close_list中，并且在open_list中删除它
            self.close_list.append(minF)
            self.open_list.remove(minF)
            # 判断这个节点的上下左右节点
            self.searchNear(minF, 0, -1, 1)
            self.searchNear(minF, 0, 1, 1)
            self.searchNear(minF, -1, 0, 1)
            self.searchNear(minF, 1, 0, 1)
            self.searchNear(minF, 1, -1, np.sqrt(2))
            self.searchNear(minF, 1, 1, np.sqrt(2))
            self.searchNear(minF, -1, -1, np.sqrt(2))
            self.searchNear(minF, -1, 1, np.sqrt(2))

            # 判断是否终止
            node = self.endPointInCloseList()

            if node:  # 如果终点在关闭表中，就返回结果
                cPoint = node
                path_list = []
                while True:
                    if cPoint.father:
                        path_list.append(cPoint.point)
                        cPoint = cPoint.father
                    else:
                        return list(reversed(path_list))

            if len(self.open_list) == 0: # 如果无法到达终点，返回空列表
                return []


    def GetPath(self):
        """
        获取该辆车辆的到目标点的路径（不考虑车辆碰撞）
        :return res_path
        """

        self.res_path = self.start()
        return self.res_path


if __name__ == "__main__":
    # 默认实际地图长度为800,比例尺 ratio = 800/栅格地图长度
    # 初始化起点终点坐标
    #[50,50] [50,175] [50,300] [50,425] [50,550]
    t0 = time.time()
    StartPoint = point.Point(50, 550)
    EndPoint   = point.Point(500, 550)
    gmap = gridmap.GridMap(32,24)
    ratio = gmap.ratio

    # 坐标转换
    startPoint = convertc.real2grid([StartPoint], ratio)[0]
    endPoint   = convertc.real2grid([EndPoint], ratio)[0]

    # 寻找路径
    astar = AStar(gmap, startPoint, endPoint)
    path = astar.GetPath()
    
    # 坐标转换
    path_real = convertc.grid2real(path, ratio)
    t1 = time.time()

    # 输出
    print('Real startPoint:', StartPoint.x, StartPoint.y ,'Grid StartPoint:', startPoint.x, startPoint.y)
    print('Real endPoint:', EndPoint.x, EndPoint.y, 'Grid EndPoint:', endPoint.x, endPoint.y)
    print('Real path:\n', [[rp[0],rp[1]] for rp in path_real])
    print('Grid path:\n', [[gp.x,gp.y] for gp in path])
    print('Consume time:', t1-t0)