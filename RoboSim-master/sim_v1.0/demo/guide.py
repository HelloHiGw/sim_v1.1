class Guide:
    def __init__(self, path=[], rotate=0, yaw=0, auto_aim=0, shoot=0):
        self.path     = path # 路径点列表
        self.rotate   = rotate # 底盘角度
        self.yaw      = yaw # 云台角度
        self.auto_aim = auto_aim # 自动瞄准
        self.shoot    = shoot # 射击
