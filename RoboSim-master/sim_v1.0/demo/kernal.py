# -*- coding: utf-8 -*-

import numpy as np
import time
import pygame
import cProfile

import car 

class bullet(object):
    def __init__(self, center, angle, speed, owner):
        self.center = center.copy()
        self.speed = speed
        self.angle = angle
        self.owner = owner


class state(object):
    def __init__(self, time, agents, compet_info, done=False, detect=None, vision=None):
        self.time   = time
        self.agents = agents
        self.compet = compet_info
        self.done   = done
        self.detect = detect
        self.vision = vision


class g_map(object):
    def __init__(self, length, width, areas, barriers):
        self.length = length
        self.width = width
        self.areas = areas
        self.barriers = barriers


class kernal(object):
    def __init__(self, car_num, time, render=False):
        self.car_num = car_num # 车辆数量
        self.time = time # 游戏时间
        self.render = render # 渲染标志
        self.guide_info = None # 决策系统的指示信息
        self.arrival = [1,1,1,1,1] # 是否到达下一个路径点:0-未到达, 1-到达
        self.next_points = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]] # 下一组路径点
        self.done = False # 游戏结束标志

        # 以下参数可根据车辆实际情况修改
        self.bullet_speed = 12.5  # 子弹速度，单位为pixel
        self.motion = 6  # 移动的惯性感大小
        self.rotate_motion = 6  # 底盘旋转的惯性感大小
        self.yaw_motion = 1  # 云台旋转的惯性感大小
        self.camera_angle = 75 / 2  # 摄像头的视野范围
        self.lidar_angle = 120 / 2  # 激光雷达的视野范围
        self.move_discount = 0.6  # 撞墙之后反弹的强度大小
        
        # 以下参数可根据实际场地情况更改
        self.map_length = 800
        self.map_width = 600

        self.theta = np.rad2deg(np.arctan(45 / 60))

        # 区域或障碍物
        self.areas = np.array([[[0.0, 100.0, 0.0, 100.0],
                                [0.0, 100.0, 125.0, 225.0],
                                [0.0, 100.0, 250.0, 350.0],
                                [0.0, 100.0, 375.0, 475.0],
                                [0.0, 100.0, 500.0, 600.0]],
                               [[700.0, 800.0, 0.0, 100.0],
                                [700.0, 800.0, 125.0, 225.0],
                                [700.0, 800.0, 250.0, 350.0],
                                [700.0, 800.0, 375.0, 475.0],
                                [700.0, 800.0, 500.0, 600.0]]], dtype='float32')

        self.barriers = np.array([[350.0, 450.0, 250.0, 275.0], # O1
                                  [350.0, 450.0, 275.0, 300.0], # O1
                                  [350.0, 450.0, 300.0, 325.0], # O1
                                  [350.0, 450.0, 325.0, 350.0], # O1
                                  [200.0, 300.0,  87.0, 113.0], # O2 -
                                  [200.0, 300.0, 487.0, 513.0], # O3 -
                                  [500.0, 600.0,  87.0, 113.0], # O4 -
                                  [500.0, 600.0, 487.0, 513.0], # O5 -
                                  [187.0, 213.0, 100.0, 200.0], # O2 |
                                  [187.0, 213.0, 400.0, 500.0], # O3 |
                                  [587.0, 613.0, 100.0, 200.0], # O4 |
                                  [587.0, 613.0, 400.0, 500.0], # O5 |
                                  ], dtype='float32')


        # 键盘控制，使用pygame作可视化
        if render:

            pygame.init()
            self.screen = pygame.display.set_mode((self.map_length, self.map_width))
            pygame.display.set_caption('Simulator')
            self.gray = (180, 180, 180)
            self.red = (190, 20, 20)
            self.blue = (10, 125, 181)

            # 加载障碍物图片
            self.barriers_img = []
            self.barriers_rect = []
            for i in range(self.barriers.shape[0]):
                self.barriers_img.append(
                    pygame.image.load('./imgs/barrier_{}.png'.format('horizontal' if i < 8 else 'vertical')))
                self.barriers_rect.append(self.barriers_img[-1].get_rect())
                self.barriers_rect[-1].center = [self.barriers[i][0:2].mean(), self.barriers[i][2:4].mean()]

            # 加载起点和车辆图片
            self.areas_img = []
            self.areas_rect = []
            for oi, o in enumerate(['red', 'blue']):
                for ti in range(5): # 起始区域
                    self.areas_img.append(pygame.image.load('./imgs/area_start_{}.png'.format(o)))
                    self.areas_rect.append(self.areas_img[-1].get_rect())
                    self.areas_rect[-1].center = [self.areas[oi, ti][0:2].mean(), self.areas[oi, ti][2:4].mean()]

            self.chassis_img = pygame.image.load('./imgs/chassis_g.png') # 底盘
            self.gimbal_img = pygame.image.load('./imgs/gimbal_g.png') # 云台
            self.bullet_img = pygame.image.load('./imgs/bullet_s.png') # 子弹
            self.bullet_rect = self.bullet_img.get_rect()
            self.info_bar_img = pygame.image.load('./imgs/info_bar.png') # 调试信息背景图片
            self.info_bar_rect = self.info_bar_img.get_rect()
            self.info_bar_rect.center = [200, self.map_width / 2]

            pygame.font.init()
            self.font = pygame.font.SysFont('info', 20)
            self.clock = pygame.time.Clock()


    def reset(self):
        """
        重置游戏参数
        :return: state(self.time, self.cars, self.compet_info, self.time <= 0)
        """

        self.time = 180
        self.orders = np.zeros((self.car_num, 8), dtype='int8')
        self.acts = np.zeros((self.car_num, 8), dtype='float32')
        self.obs = np.zeros((self.car_num, 17), dtype='float32')
        self.compet_info = np.array([[2, 1, 0, 0], [2, 1, 0, 0]], dtype='int16')
        self.vision = np.zeros((self.car_num, self.car_num), dtype='int8')
        self.detect = np.zeros((self.car_num, self.car_num), dtype='int8')
        self.bullets = []
        self.epoch = 0
        self.n = 0
        self.dev = False
        self.memory = []
        self.done = False

        # 定义车的初始状态
        # [team, x, y, yaw, yaw_PTZ, heat, HP, freeze_time, Issupply, can_shoot, bullet, stay_time?, wheel_hit, armor_hit, car_hit]
        cars = np.array([[0, 50,  50, 0, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [0, 50, 175, 0, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [0, 50, 300, 0, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [0, 50, 425, 0, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [0, 50, 550, 0, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],

                         [1, 750,  50, 180, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [1, 750, 175, 180, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [1, 750, 300, 180, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [1, 750, 425, 180, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0],
                         [1, 750, 550, 180, 0, 0, 2000, 0, 0, 1, 100, 0, 0, 0, 0]], dtype='float32')

        self.cars = cars[0:self.car_num]
        print('reset')
        print(self.time)
        print(self.cars[0:, 6])
        return state(self.time, self.cars, self.compet_info, self.time <= 0)


    def play(self, cars_info, guide_info, endFlag):
        """
        开始游戏（接收指令并刷新）
        :return: None
        """

        # 手动模式，render必须为True
        assert self.render, 'human play mode, only when render == True'
        while True:
            if not self.epoch % 10: # 每10个epoch接收1次命令
                if self.get_order() :
                    endFlag.put(1) # 窗口被关闭
                    break
            self.one_epoch(cars_info, guide_info)


    def step(self, orders):
        """
        单步运行
        :param orders: 指令参数
        :return: state(self.time, self.cars, self.compet_info, self.time <= 0, self.detect, self.vision)
        """

        self.orders = orders

        for _ in range(10): # 每个step运行10个epoch
            self.one_epoch()

        return state(self.time, self.cars, self.compet_info, self.done, self.detect, self.vision)


    def one_epoch(self):
        """
        新回合刷新
        :return: None
        """

        for n in range(self.car_num): # 循环更改车辆参数
            if self.cars[n, 6] <=0:  # 被击败的车辆略过
                self.cars[n, 6] = 0
                continue
            if not self.epoch % 10:
                self.orders_to_acts(n)

                """
                # 获取最新指示信息
                if guide_info.empty():
                    pass
                else:
                    self.guide_info = guide_info.get()
                    print('guide_info: ', self.guide_info)
                self.get_acts(self.guide_info, 0, cars_info) 
                # 更新车辆信息
                if not cars_info.empty(): # 清除之前未使用的数据
                    _ = cars_info.get()
                cars_info.put(self.get_car_info()) 
                """

            # 移动车辆
            self.move_car(n)
            if not self.acts[n, 5]: self.acts[n, 4] = 0

        if not self.epoch % 200: # 200个epoch为1个单位时间(即1s)
            self.time -= 1
        self.get_camera_vision()

        # 移动子弹
        i = 0
        while len(self.bullets):
            if self.move_bullet(i):
                del self.bullets[i]
                i -= 1
            i += 1
            if i >= len(self.bullets): break
        bullets = []
        for i in range(len(self.bullets)):
            bullets.append(bullet(self.bullets[i].center, self.bullets[i].angle, self.bullets[i].speed, self.bullets[i].owner))
        if self.render: 
            for event in pygame.event.get(): # 获取事件但不处理
                pass
            self.update_display()

        # 比赛结束检查 
        if self.time <= 0 or (self.cars[0:5, 6] == np.zeros(5)).all() or (self.cars[5:, 6] == np.zeros(5)).all():
            print('check game end')
            print(self.time)
            print(self.cars[0:, 6])
            self.done = True

        self.epoch += 1


    def move_car(self, n): # jitable
        """
        移动车辆
        :param n: 车辆编号 0-9
        :return: None
        """

        # move chassis
        if self.acts[n, 0]:
            p = self.cars[n, 3]
            self.cars[n, 3] += self.acts[n, 0]
            if self.cars[n, 3] > 180: self.cars[n, 3] -= 360
            if self.cars[n, 3] < -180: self.cars[n, 3] += 360
            if self.check_interface(n):
                self.acts[n, 0] = -self.acts[n, 0] * self.move_discount
                self.cars[n, 3] = p

        # move gimbal
        if self.acts[n, 1]:
            self.cars[n, 4] += self.acts[n, 1]
            if self.cars[n, 4] > 90: self.cars[n, 4] = 90
            if self.cars[n, 4] < -90: self.cars[n, 4] = -90

        if self.acts[n, 7]:
            select = np.where((self.vision[n] == 1))[0]
            if select.size:
                angles = np.zeros(select.size)
                for ii, i in enumerate(select):
                    if cars[n, 0] == cars[i, 0]: # 己方车辆跳过自瞄检测
                        continue
                    x, y = self.cars[i, 1:3] - self.cars[n, 1:3]
                    angle = np.angle(x + y * 1j, deg=True) - self.cars[i, 3]
                    if angle >= 180: angle -= 360
                    if angle <= -180: angle += 360
                    if angle >= -self.theta and angle < self.theta:
                        armor = self.get_armor(self.cars[i], 2)
                    elif angle >= self.theta and angle < 180 - self.theta:
                        armor = self.get_armor(self.cars[i], 3)
                    elif angle >= -180 + self.theta and angle < -self.theta:
                        armor = self.get_armor(self.cars[i], 1)
                    else:
                        armor = self.get_armor(self.cars[i], 0)
                    x, y = armor - self.cars[n, 1:3]
                    angle = np.angle(x + y * 1j, deg=True) - self.cars[n, 4] - self.cars[n, 3]
                    if angle >= 180: angle -= 360
                    if angle <= -180: angle += 360
                    angles[ii] = angle
                m = np.where(np.abs(angles) == np.abs(angles).min())
                self.cars[n, 4] += angles[m][0]
                if self.cars[n, 4] > 90: self.cars[n, 4] = 90
                if self.cars[n, 4] < -90: self.cars[n, 4] = -90

        # move x and y
        if self.acts[n, 2] or self.acts[n, 3]:
            angle = np.deg2rad(self.cars[n, 3])
            # x
            p = self.cars[n, 1]
            self.cars[n, 1] += (self.acts[n, 2]) * np.cos(angle) - (self.acts[n, 3]) * np.sin(angle)
            if self.check_interface(n):
                self.acts[n, 2] = -self.acts[n, 2] * self.move_discount
                self.cars[n, 1] = p
            # y
            p = self.cars[n, 2]
            self.cars[n, 2] += (self.acts[n, 2]) * np.sin(angle) + (self.acts[n, 3]) * np.cos(angle)
            if self.check_interface(n):
                self.acts[n, 3] = -self.acts[n, 3] * self.move_discount
                self.cars[n, 2] = p

        # fire or not
        if self.acts[n, 4] and self.cars[n, 10]:
            if self.cars[n, 9]:
                self.cars[n, 10] -= 1
                self.bullets.append(
                    bullet(self.cars[n, 1:3], self.cars[n, 4] + self.cars[n, 3], self.bullet_speed, n))
                self.cars[n, 5] += self.bullet_speed
                self.cars[n, 9] = 0
            else:
                self.cars[n, 9] = 1
        else:
            self.cars[n, 9] = 1

        # check supply
        if self.acts[n, 6]:
            dis = np.abs(self.cars[n, 1:3] - [self.areas[int(self.cars[n, 0]), 1][0:2].mean(),
                                              self.areas[int(self.cars[n, 0]), 1][2:4].mean()]).sum()
            if dis < 23 and self.compet_info[int(self.cars[n, 0]), 0] and not self.cars[n, 7]:
                self.cars[n, 8] = 1
                self.cars[n, 7] = 600  # 3 s
                self.cars[n, 10] += 50
                self.compet_info[int(self.cars[n, 0]), 0] -= 1


    def move_bullet(self, n): # jitable
        """
        移动子弹并进行扣血检查
        :param n: 子弹编号 0-9
        return Bool
        """

        old_point = self.bullets[n].center.copy()
        self.bullets[n].center[0] += self.bullets[n].speed * np.cos(np.deg2rad(self.bullets[n].angle))
        self.bullets[n].center[1] += self.bullets[n].speed * np.sin(np.deg2rad(self.bullets[n].angle))

        # 子弹出界
        if self.bullets[n].center[0] <= 0 or self.bullets[n].center[0] >= self.map_length \
                or self.bullets[n].center[1] <= 0 or self.bullets[n].center[1] >= self.map_width: 
                return True

        # 子弹撞到障碍物
        for b in self.barriers:
            if self.line_barriers_check(self.bullets[n].center, old_point): 
                return True

        # 子弹击打装甲板
        for i in range(len(self.cars)):
            if i == self.bullets[n].owner: 
                continue
            if np.abs(np.array(self.bullets[n].center) - np.array(self.cars[i, 1:3])).sum() < 52.5:
                points = self.transfer_to_car_coordinate(np.array([self.bullets[n].center, old_point]), i)
                if self.segment(points[0], points[1], [-18.5, -5], [-18.5, 6]) \
                        or self.segment(points[0], points[1], [18.5, -5], [18.5, 6]) \
                        or self.segment(points[0], points[1], [-5, 30], [5, 30]) \
                        or self.segment(points[0], points[1], [-5, -30], [5, -30]):
                    if self.compet_info[int(self.cars[i, 0]), 3]:
                        self.cars[i, 6] -= 25
                    else:
                        self.cars[i, 6] -= 50
                    return True
                if self.line_rect_check(points[0], points[1], [-18, -29, 18, 29]): 
                    return True
        return False


    def update_display(self):
        """
        更新显示
        :return: None
        """

        assert self.render, 'only render mode need update_display'
        self.screen.fill(self.gray)
        for i in range(len(self.barriers_rect)): # 渲染障碍物
            self.screen.blit(self.barriers_img[i], self.barriers_rect[i])

        for i in range(len(self.areas_rect)): # 渲染区域
            self.screen.blit(self.areas_img[i], self.areas_rect[i])

        for i in range(len(self.bullets)): # 渲染子弹
            self.bullet_rect.center = self.bullets[i].center
            self.screen.blit(self.bullet_img, self.bullet_rect)

        for n in range(self.car_num): # 渲染车辆
            chassis_rotate = pygame.transform.rotate(self.chassis_img, -self.cars[n, 3] - 90)
            gimbal_rotate = pygame.transform.rotate(self.gimbal_img, -self.cars[n, 4] - self.cars[n, 3] - 90)
            chassis_rotate_rect = chassis_rotate.get_rect()
            gimbal_rotate_rect = gimbal_rotate.get_rect()
            chassis_rotate_rect.center = self.cars[n, 1:3]
            gimbal_rotate_rect.center = self.cars[n, 1:3]
            self.screen.blit(chassis_rotate, chassis_rotate_rect)
            self.screen.blit(gimbal_rotate, gimbal_rotate_rect)
        # self.screen.blit(self.head_img[0], self.head_rect[0])
        # self.screen.blit(self.head_img[1], self.head_rect[1])

        for n in range(self.car_num): # 渲染车辆
            select = np.where((self.vision[n] == 1))[0] + 1
            select2 = np.where((self.detect[n] == 1))[0] + 1
            info = self.font.render('{} | {}: {} {}'.format(int(self.cars[n, 6]), n + 1, select, select2), True,
                                    self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3] + [-20, -60])
            info = self.font.render('{} {}'.format(int(self.cars[n, 10]), int(self.cars[n, 5])), True,
                                    self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3] + [-20, -45])
        info = self.font.render('time: {}'.format(self.time), False, (0, 0, 0))
        # self.screen.blit(info, (8, 8))

        if 0: self.dev_window()
        pygame.display.update()
        self.clock.tick(40)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))


    def dev_window(self):
        """
        渲染开发者窗口
        :return: None
        """
        for n in range(self.car_num): # 显示车辆轮廓点
            wheels = self.check_points_wheel(self.cars[n])
            for w in wheels:
                pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, w.astype(int), 3)
            armors = self.check_points_armor(self.cars[n])
            for a in armors:
                pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, a.astype(int), 3)
        self.screen.blit(self.info_bar_img, self.info_bar_rect)

        for n in range(self.car_num//2): # 显示车辆详细信息
            tags = ['owner', 'x', 'y', 'angle', 'yaw', 'heat', 'hp', 'freeze_time', 'is_supply',
                    'can_shoot', 'bullet', 'stay_time', 'wheel_hit', 'armor_hit', 'car_hit']
            info = self.font.render('car {}'.format(n), False, (0, 0, 0))
            self.screen.blit(info, (8 + n * 100, 100))
            for i in range(self.cars[n].size):
                info = self.font.render('{}: {}'.format(tags[i], int(self.cars[n, i])), False, (0, 0, 0))
                self.screen.blit(info, (8 + n * 100, 117 + i * 17))

        for n in range(self.car_num//2): # 显示车辆详细信息
            tags = ['owner', 'x', 'y', 'angle', 'yaw', 'heat', 'hp', 'freeze_time', 'is_supply',
                    'can_shoot', 'bullet', 'stay_time', 'wheel_hit', 'armor_hit', 'car_hit']
            info = self.font.render('car {}'.format(n+5), False, (0, 0, 0))
            self.screen.blit(info, (8 + n * 100, 100))
            for i in range(self.cars[n+5].size):
                info = self.font.render('{}: {}'.format(tags[i], int(self.cars[n+5, i])), False, (0, 0, 0))
                self.screen.blit(info, (8 + n * 100, 467 + i * 17))

        info = self.font.render('red   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[0, 0], self.compet_info[0, 1], self.compet_info[0, 3]), False, (0, 0, 0))
        self.screen.blit(info, (8, 372))
        info = self.font.render('blue   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[1, 0], self.compet_info[1, 1], self.compet_info[1, 3]), False,(0, 0, 0))
        self.screen.blit(info, (8, 389))


    def get_order(self):
        """
        获取车辆指令
        :return: Bool
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_0]: self.n = 0
        if pressed[pygame.K_1]: self.n = 1
        if pressed[pygame.K_2]: self.n = 2
        if pressed[pygame.K_3]: self.n = 3
        if pressed[pygame.K_4]: self.n = 4
        if pressed[pygame.K_5]: self.n = 5
        if pressed[pygame.K_6]: self.n = 6
        if pressed[pygame.K_7]: self.n = 7
        if pressed[pygame.K_8]: self.n = 8
        if pressed[pygame.K_9]: self.n = 9

        self.orders[:] = 0

        if pressed[pygame.K_w]: self.orders[self.n, 0] += 1
        if pressed[pygame.K_s]: self.orders[self.n, 0] -= 1
        if pressed[pygame.K_q]: self.orders[self.n, 1] -= 1
        if pressed[pygame.K_e]: self.orders[self.n, 1] += 1
        if pressed[pygame.K_a]: self.orders[self.n, 2] -= 1
        if pressed[pygame.K_d]: self.orders[self.n, 2] += 1
        if pressed[pygame.K_b]: self.orders[self.n, 3] -= 1
        if pressed[pygame.K_m]: self.orders[self.n, 3] += 1
        if pressed[pygame.K_SPACE]:
            self.orders[self.n, 4] = 1
        else:
            self.orders[self.n, 4] = 0
        if pressed[pygame.K_f]:
            self.orders[self.n, 5] = 1
        else:
            self.orders[self.n, 5] = 0
        if pressed[pygame.K_r]:
            self.orders[self.n, 6] = 1
        else:
            self.orders[self.n, 6] = 0
        if pressed[pygame.K_n]:
            self.orders[self.n, 7] = 1
        else:
            self.orders[self.n, 7] = 0

        if pressed[pygame.K_TAB]:
            self.dev = True
        else:
            self.dev = False
        return False

        '''

        self.orders[:] = 0
        # get orders from joysticks
        pygame.joystick.init()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

        joystick_count = pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()

            # axis
            self.orders[i, 1] -= round(joystick.get_axis(0))  # 左右，q/e，左摇杆
            self.orders[i, 0] -= round(joystick.get_axis(1))  # 前后，w/s，左摇杆
            self.orders[i, 2] += round(joystick.get_axis(4))  # 底盘旋转，a/d，LT/RT
            self.orders[i, 3] -= round(joystick.get_axis(2))  # 云台旋转，b/m，右摇杆

            # button
            self.orders[i, 4] = joystick.get_button(5)  # 射击，space，RB键
            self.orders[i, 5] = joystick.get_button(4)  # 补给，f，LB键
            self.orders[i, 6] = joystick.get_button(3)  # 连发，r，Y键
            self.orders[i, 7] = joystick.get_button(0)  # 自瞄，n，A键

            # hat
            self.orders[i, 1] += round(joystick.get_hat(0)[0])  # 左右，q/e
            self.orders[i, 0] += round(joystick.get_hat(0)[1])  # 前后，w/s
            np.round(self.orders)

            if joystick.get_button(2):
                self.dev = True
            else:
                self.dev = False

        # return False
        '''


    def get_acts(self, guide_info, team, cars_info):
        """
        根据指示信息控制车辆
        :param guide_info: 5辆车的指示信息
        :param team: 队伍编号
        :return: None
        """

        v_max = 1 # 最大速度

        if len(guide_info) != 5:
            print('No five paths for all cars!')
            return
        if team == 0:
            car_pos = self.cars[0:5, 1:3]
            car_rotate = self.cars[0:5, 3]
        elif team == 1:
            car_pos = self.cars[5:10, 1:3]
            car_rotate = self.cars[5:10, 3]
        else:
            print('Team number is wrong!')
            return

        # 路径点跟踪、自瞄与射击控制
        for i in range(5):
            if self.arrival[i] == 1:
                if len(guide_info[i].path) != 0: # 获取路径的第一个点并从路径中删去
                    self.next_points[i] = guide_info[i].path[0]
                    del guide_info[i].path[0]
                    self.arrival[i] = 0

                    x_dis = self.next_points[i][0] - car_pos[i][0]
                    y_dis = self.next_points[i][1] - car_pos[i][1]

                    if abs(x_dis)<5 and abs(y_dis)<5: # 到达该路径点时停止运动,更新标志位
                        self.acts[i][2] = 0
                        self.acts[i][3] = 0
                        self.arrival[i] = 1
                    else:
                        rotate = car_rotate[i]
                        dis = (x_dis**2 + y_dis**2) ** 0.5
                        v_X = v_max * x_dis / dis
                        v_Y = v_max * y_dis / dis
                        # print(i, 'v_X:', v_X, 'v_Y:', v_Y)
                        self.acts[i][2] = v_X * np.cos(np.deg2rad(rotate)) + v_Y * np.sin(np.deg2rad(rotate))
                        self.acts[i][3] = v_X * -np.sin(np.deg2rad(rotate)) + v_Y * np.cos(np.deg2rad(rotate))
                        # print(i, 'x_speed:', x_speed, 'y_speed:', y_speed, '\n')
                else:
                    print('num', i, 'No path')
            else:
                print('num', i, 'Moving')
                x_dis = self.next_points[i][0] - car_pos[i][0]
                y_dis = self.next_points[i][1] - car_pos[i][1]

                if abs(x_dis)<5 and abs(y_dis)<5: # 到达该路径点时停止运动,更新标志位
                    self.acts[i][2] = 0
                    self.acts[i][3] = 0
                    self.arrival[i] = 1
                else:
                    rotate = car_rotate[i]
                    dis = (x_dis**2 + y_dis**2) ** 0.5
                    v_X = v_max * x_dis / dis
                    v_Y = v_max * y_dis / dis
                    # print(i, 'v_X:', v_X, 'v_Y:', v_Y)
                    self.acts[i][2] = v_X * np.cos(np.deg2rad(rotate)) + v_Y * np.sin(np.deg2rad(rotate))
                    self.acts[i][3] = v_X * -np.sin(np.deg2rad(rotate)) + v_Y * np.cos(np.deg2rad(rotate))
                    # print(i, 'x_speed:', x_speed, 'y_speed:', y_speed, '\n')
            print('car num:',i,'next point:',self.next_points[i])


    def orders_to_acts(self, n):
        """
        将指令转化为动作
        :param n: 车辆编号 0-9
        :return: None
        """

        # x
        self.acts[n, 2] += self.orders[n, 0] * 1.5 / self.motion
        if self.orders[n, 0] == 0: # when x = 0, x_speed motion
            if self.acts[n, 2] > 0: self.acts[n, 2] -= 1.5 / self.motion
            if self.acts[n, 2] < 0: self.acts[n, 2] += 1.5 / self.motion
        if abs(self.acts[n, 2]) < 1.5 / self.motion: self.acts[n, 2] = 0
        if self.acts[n, 2] >= 1.5: self.acts[n, 2] = 1.5
        if self.acts[n, 2] <= -1.5: self.acts[n, 2] = -1.5

        # y
        self.acts[n, 3] += self.orders[n, 1] * 1 / self.motion
        if self.orders[n, 1] == 0: # when y = 0, y_speed motion
            if self.acts[n, 3] > 0: self.acts[n, 3] -= 1 / self.motion
            if self.acts[n, 3] < 0: self.acts[n, 3] += 1 / self.motion
        if abs(self.acts[n, 3]) < 1 / self.motion: self.acts[n, 3] = 0
        if self.acts[n, 3] >= 1: self.acts[n, 3] = 1
        if self.acts[n, 3] <= -1: self.acts[n, 3] = -1

        # rotate
        self.acts[n, 0] += self.orders[n, 2] * 1 / self.rotate_motion
        if self.orders[n, 2] == 0: # when rotate = 0, rotate_speed motion
            if self.acts[n, 0] > 0: self.acts[n, 0] -= 1 / self.rotate_motion
            if self.acts[n, 0] < 0: self.acts[n, 0] += 1 / self.rotate_motion
        if abs(self.acts[n, 0]) < 1 / self.rotate_motion: self.acts[n, 0] = 0
        if self.acts[n, 0] > 1: self.acts[n, 0] = 1
        if self.acts[n, 0] < -1: self.acts[n, 0] = -1

        # yaw_PTZ
        self.acts[n, 1] += self.orders[n, 3] / self.yaw_motion
        if self.orders[n, 3] == 0: # when yaw = 0, yaw_speed motion
            if self.acts[n, 1] > 0: self.acts[n, 1] -= 1 / self.yaw_motion
            if self.acts[n, 1] < 0: self.acts[n, 1] += 1 / self.yaw_motion
        if abs(self.acts[n, 1]) < 1 / self.yaw_motion: self.acts[n, 1] = 0
        if self.acts[n, 1] > 3: self.acts[n, 1] = 3
        if self.acts[n, 1] < -3: self.acts[n, 1] = -3

        self.acts[n, 4] = self.orders[n, 4]
        self.acts[n, 6] = self.orders[n, 5]
        self.acts[n, 5] = self.orders[n, 6]
        self.acts[n, 7] = self.orders[n, 7]


    def set_car_loc(self, n, loc):
        """
        设定车辆位置
        :param n: 车辆编号 0-9
        :param loc: 指定位置和航向角
        :return: None
        """

        self.cars[n, 1:3] = loc


    def get_map(self):
        """
        获取地图信息
        :return: g_map(self.map_length, self.map_width, self.areas, self.barriers)
        """

        return g_map(self.map_length, self.map_width, self.areas, self.barriers)


    def cross(self, p1, p2, p3): # jitable
        """
        叉积判断
        :param p1: 点1坐标
        :param p2: 点2坐标
        :param p3: 点3坐标
        :return x1 * y2 - x2 * y1
        """
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a

        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1


    def segment(self, p1, p2, p3, p4): #jitable
        """
        判断两条线段是否相交
        :param p1: 点1坐标
        :param p2: 点2坐标
        :param p3: 点3坐标
        :param p4: 点4坐标
        :return: Bool
        """

        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        if (max(p1[0], p2[0]) >= min(p3[0], p4[0]) \
        and max(p3[0], p4[0]) >= min(p1[0], p2[0]) \
        and max(p1[1], p2[1]) >= min(p3[1], p4[1]) \
        and max(p3[1], p4[1]) >= min(p1[1], p2[1])): # 矩形相交判断
            if (self.cross(p1, p2, p3) * self.cross(p1, p2, p4) <= 0 \
            and self.cross(p3, p4, p1) * self.cross(p3, p4, p2) <= 0): # 叉积判断
                return True
            else:
                return False
        else:
            return False


    def line_rect_check(self, l1, l2, sq): # jitable
        """
        判定线段与矩形是否相交
        :param l1: 线段点1坐标
        :param l2: 线段点2坐标
        :param sq: 矩形坐标
        :return: Bool
        """
        
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        # check if line cross rect, sq = [x_leftdown, y_leftdown, x_rightup, y_rightup]
        p1 = [sq[0], sq[1]]
        p2 = [sq[2], sq[3]]
        p3 = [sq[2], sq[1]]
        p4 = [sq[0], sq[3]]
        if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4):
            return True
        else:
            return False


    def line_barriers_check(self, l1, l2): # jitable
        """
        检测两车连线上是否有障碍物
        :param l1: 线段点1坐标
        :param l2: 线段点2坐标
        :return: Bool
        """

        for b in self.barriers:
            sq = [b[0], b[2], b[1], b[3]]  # [left, up, right, down]
            if self.line_rect_check(l1, l2, sq):  return True
        return False


    def line_cars_check(self, l1, l2): 
        """
        检测两车连线上是否有其他机器人
        :param l1: 线段点1坐标
        :param l2: 线段点2坐标
        :return: Bool
        """

        for car in self.cars:
            if (car[1:3] == l1).all() or (car[1:3] == l2).all():
                continue
            p1, p2, p3, p4 = self.get_car_outline(car)
            if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4): return True
        return False


    def get_lidar_vision(self):
        """
        获取雷达视觉范围
        :return: None
        """

        for n in range(self.car_num):
            for i in range(self.car_num - 1):
                if self.cars[n, 0] == self.cars[n-i-1, 0]:
                    continue
                x, y = self.cars[n - i - 1, 1:3] - self.cars[n, 1:3]
                angle = np.angle(x + y * 1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angle = angle - self.cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if abs(angle) < self.lidar_angle:
                    if self.line_barriers_check(self.cars[n, 1:3], self.cars[n - i - 1, 1:3]) \
                            or self.line_cars_check(self.cars[n, 1:3], self.cars[n - i - 1, 1:3]):
                        self.detect[n, n - i - 1] = 0
                    else:
                        self.detect[n, n - i - 1] = 1
                else:
                    self.detect[n, n - i - 1] = 0


    def get_camera_vision(self):
        """
        获取相机视觉范围
        :return: None
        """

        for n in range(self.car_num):
            for i in range(self.car_num - 1):
                if self.cars[n, 0] == self.cars[n-i-1, 0]:
                    continue
                x, y = self.cars[n - i - 1, 1:3] - self.cars[n, 1:3]
                angle = np.angle(x + y * 1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angle = angle - self.cars[n, 4] - self.cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if abs(angle) < self.camera_angle: # 是否在相机视角内
                    if self.line_barriers_check(self.cars[n, 1:3], self.cars[n - i - 1, 1:3]) \
                            or self.line_cars_check(self.cars[n, 1:3], self.cars[n - i - 1, 1:3]): 
                        self.vision[n, n - i - 1] = 0
                    else:
                        self.vision[n, n - i - 1] = 1
                else:
                    self.vision[n, n - i - 1] = 0


    def transfer_to_car_coordinate(self, points, n): # jitable
        """
        :param points: 待转换点坐标
        :param n: 车辆编号 0-9
        全局坐标系到车辆坐标系的转换
        :return: np.matmul(points + pan_vecter, rotate_matrix)
        """

        pan_vecter = -self.cars[n, 1:3]
        rotate_matrix = np.array([[np.cos(np.deg2rad(self.cars[n, 3] + 90)), -np.sin(np.deg2rad(self.cars[n, 3] + 90))],
                                  [np.sin(np.deg2rad(self.cars[n, 3] + 90)), np.cos(np.deg2rad(self.cars[n, 3] + 90))]])
        return np.matmul(points + pan_vecter, rotate_matrix)


    def check_points_wheel(self, car):
        """
        检查车轮点
        :param car: 车辆信息
        :return: [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]
        """

        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                  [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]])
        xs = np.array([[-22.5, -29], [22.5, -29],
                       [-22.5, -14], [22.5, -14],
                       [-22.5, 14], [22.5, 14],
                       [-22.5, 29], [22.5, 29]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]


    def check_points_armor(self, car):
        """
        检查装甲点
        :param car: 车辆信息
        :return: [np.matmul(x, rotate_matrix) + car[1:3] for x in xs]
        """

        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                  [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]])
        xs = np.array([[-6.5, -30], [6.5, -30],
                       [-18.5, -7], [18.5, -7],
                       [-18.5, 0], [18.5, 0],
                       [-18.5, 6], [18.5, 6],
                       [-6.5, 30], [6.5, 30]])
        return [np.matmul(x, rotate_matrix) + car[1:3] for x in xs]


    def get_car_outline(self, car): # jitable
        """
        检查车辆轮廓
        :param car: 车辆信息
        :return: [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]

        """

        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                  [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]])
        xs = np.array([[-22.5, -30], [22.5, 30], [-22.5, 30], [22.5, -30]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]


    def check_interface(self, n): # jitable
        """
        检查车辆表面
        :param n: 车辆编号
        :return: Bool
        """

        # car barriers assess
        wheels = self.check_points_wheel(self.cars[n])
        for w in wheels:
            if w[0] <= 0 or w[0] >= self.map_length or w[1] <= 0 or w[1] >= self.map_width:
                self.cars[n, 12] += 1
                return True
            for b in self.barriers:
                if w[0] >= b[0] and w[0] <= b[1] and w[1] >= b[2] and w[1] <= b[3]:
                    self.cars[n, 12] += 1
                    return True
        armors = self.check_points_armor(self.cars[n])
        for a in armors:
            if a[0] <= 0 or a[0] >= self.map_length or a[1] <= 0 or a[1] >= self.map_width:
                self.cars[n, 13] += 1
                self.cars[n, 6] -= 10
                return True
            for b in self.barriers:
                if a[0] >= b[0] and a[0] <= b[1] and a[1] >= b[2] and a[1] <= b[3]:
                    self.cars[n, 13] += 1
                    self.cars[n, 6] -= 10
                    return True

        # car car assess
        for i in range(self.car_num):
            if i == n: continue
            wheels_tran = self.transfer_to_car_coordinate(wheels, i)
            for w in wheels_tran:
                if w[0] >= -22.5 and w[0] <= 22.5 and w[1] >= -30 and w[1] <= 30:
                    self.cars[n, 14] += 1
                    return True
            armors_tran = self.transfer_to_car_coordinate(armors, i)
            for a in armors_tran:
                if a[0] >= -22.5 and a[0] <= 22.5 and a[1] >= -30 and a[1] <= 30:
                    self.cars[n, 14] += 1
                    self.cars[n, 6] -= 10
                    return True
        return False


    def get_armor(self, car, i):
        """
        获取装甲位置信息
        :param car: 车辆信息 
        :param: i: 装甲编号 0-3
        :return: np.matmul(xs[i], rotate_matrix) + car[1:3]
        """

        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                  [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]])
        xs = np.array([[0, -30], [18.5, 0], [0, 30], [-18.5, 0]])
        return np.matmul(xs[i], rotate_matrix) + car[1:3]


    def get_car_info(self):
        """
        返回所有车辆信息
        :return: cars_information
        """

        cars = self.cars
        cars_infomation = []

        for i in range(10):
            curCar = car.Car(cars[i,0], cars[i,1], cars[i,2], cars[i,3], cars[i,4], cars[i,6], cars[i,10])
            cars_infomation.append(curCar)

        return cars_infomation


def test():
    game = kernal(10, 180, render=False)
    game.reset()

    fake_orders = np.ones((10,8), dtype='int8')
    for j in range(1):
        print('episode:', j)
        for i in range(10):
            step_info = game.step(fake_orders)
            print('step:', i, '-', step_info)
            print(game.time)
            game.time -= 5
        game.reset()


if __name__ == "__main__": 
    cProfile.run('test()')