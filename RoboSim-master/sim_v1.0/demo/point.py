import sys

class Point:
    def __init__(self, x, y):
        self.x = x # 地图点x坐标
        self.y = y # 地图点y坐标
    
    def __eq__(self, other):
        if other == None:
            return False
        if self.x == other.x and self.y == other.y:
            return True
        return False