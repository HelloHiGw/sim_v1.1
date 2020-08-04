# -*- coding: utf-8 -*-

import numpy as np
import pygame
from numba import jit

import cProfile
import timeit


"""
# constant params illustration

# playground params
map_length    = 800   # length int32
map_width     = 600   # width int32

# car params
car_num       = 10    # num of cars int32
bullet_speed  = 12.5  # bullet speed, pixel/epoch float32
motion        = 6     # move motion float32
rotate_motion = 6     # chasis rotate motion  float32
yaw_motion    = 1     # gimbal rotate motion float32
camera_angle  = 75 / 2   # angle of camera float32
lidar_angle   = 120 / 2  # angle of lidar float32
move_discount = 0.6   #  magnitude of rebound when hit border/barrier/car float32
theta         = np.rad2deg(np.arctan(45 / 60)) # angle of autoaim float32

# map params
areas         = np.array([[[0.0, 100.0, 0.0, 100.0],
                            [0.0, 100.0, 125.0, 225.0],
                            [0.0, 100.0, 250.0, 350.0],
                            [0.0, 100.0, 375.0, 475.0],
                            [0.0, 100.0, 500.0, 600.0]],
                            [[700.0, 800.0, 0.0, 100.0],
                            [700.0, 800.0, 125.0, 225.0],
                            [700.0, 800.0, 250.0, 350.0],
                            [700.0, 800.0, 375.0, 475.0],
                            [700.0, 800.0, 500.0, 600.0]]], 
                            dtype=np.float32) # start areas float32-2x5x4
barriers      = np.array([[350.0, 450.0, 250.0, 275.0], # O1
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
                            ], dtype=np.float32) # barriers float32-ax4
"""


# type define
bullet_type = np.dtype({
    'names': ['x', 'y', 'speed', 'angle', 'owner', 'state'],
    'formats': [np.float32, np.float32, np.float32, np.float32, np.int32, np.int32],
}) # state: 0-to be fired, 1-fly, 2-Miss(hit border/barrier), 10-19-hit(armor of car 0-9) 


@jit(nopython=True)
def init():
    """
    Initialize all the variables
    :return: time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts, 
            guides, arrivals, next_points, control_n
    """

    car_num = 10

    time    = np.array([180], dtype=np.int32)   # time of game int32
    done    = np.array([0], dtype=np.int8)      # if game end int8
    epoch   = np.array([0], dtype=np.int32)     # rounds of update int32
    detect  = np.zeros((10, 10), dtype=np.int8) # lidar detection matrix int8-10x10
    vision  = np.zeros((10, 10), dtype=np.int8) # camera detection matrix int8-10x10

    # [team, x, y, rotate, yaw, HP, can_shoot, bullet, wheel_hit, armor_hit, car_hit]
    cars    = np.array([[0, 50,  50, 0, 0, 2000, 1, 100, 0, 0, 0],
                    [0, 50, 175, 0, 0, 2000, 1, 100, 0, 0, 0],
                    [0, 50, 300, 0, 0, 2000, 1, 100, 0, 0, 0],
                    [0, 50, 425, 0, 0, 2000, 1, 100, 0, 0, 0],
                    [0, 50, 550, 0, 0, 2000, 1, 100, 0, 0, 0],

                    [1, 750,  50, 180, 0, 2000, 1, 100, 0, 0, 0],
                    [1, 750, 175, 180, 0, 2000, 1, 100, 0, 0, 0],
                    [1, 750, 300, 180, 0, 2000, 1, 100, 0, 0, 0],
                    [1, 750, 425, 180, 0, 2000, 1, 100, 0, 0, 0],
                    [1, 750, 550, 180, 0, 2000, 1, 100, 0, 0, 0]], 
                    dtype=np.float32)           # car states float32-10x11
    bullets = np.zeros(1000, dtype=bullet_type) # bullet states
    bullets_count = np.zeros(10, dtype=np.int32) # counts of bullets that fired

    orders  = np.zeros((car_num, 6), dtype=np.int8)    # control order int8-10x6
    acts    = np.zeros((car_num, 6), dtype=np.float32) # bottom control float32-10x6
    guides  = np.array([[-1,-1,0,-1],
                        [-1,-1,0,-1],
                        [-1,-1,0,-1],
                        [-1,-1,0,-1],
                        [-1,-1,0,-1]], 
                        dtype=np.int32)           # guides from decision-making system [x, y, ifShoot, targetNum] int32-5x4
    arrivals    = np.ones(5, dtype=np.int8)       # if cars arrive at target position: 0-not, 1-yes int8-5
    next_points = np.zeros((5,2), dtype=np.int32) # next group of target position int32-5x2
    control_n   = -np.ones(1, dtype=np.int32)     # num of car that controled by keyboard int32

    return time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts, \
            guides, arrivals, next_points, control_n


@jit(nopython=True)
def reset(time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts, 
            guides, arrivals, next_points, control_n):
    """
    Reset all the variables
    :return: None
    """

    time[0]  = 180
    done[0]  = 0
    epoch[0] = 0
    for i in range(10):
        for j in range(10):
            detect[i][j] = 0
            vision[i][j] = 0

    for i in range(5):
        cars[i][0] = 0
        cars[i][1] = 50
        cars[i][2] = 50 + i * 125
        cars[i][3] = 0
        cars[i][4] = 0
        cars[i][5] = 2000
        cars[i][6] = 1
        cars[i][7] = 100
        cars[i][8] = 0
        cars[i][9] = 0
        cars[i][10] = 0
    
    for i in range(5,10):
        cars[i][0] = 1
        cars[i][1] = 750
        cars[i][2] = 50 + (i-5)*125
        cars[i][3] = 180
        cars[i][4] = 0
        cars[i][5] = 2000
        cars[i][6] = 1
        cars[i][7] = 100
        cars[i][8] = 0
        cars[i][9] = 0
        cars[i][10] = 0
    
    for i in range(1000):
        bullets[i]['x'] = 0.0
        bullets[i]['y'] = 0.0
        bullets[i]['angle'] = 0.0
        bullets[i]['speed'] = 0.0
        bullets[i]['owner'] = 0
        bullets[i]['state'] = 0

    for i in range(10):
        bullets_count[i] = 0

    for i in range(10):
        for j in range(6):
            orders[i][j] = 0
            acts[i][j]   = 0

    for i in range(5):
        guides[i][0] = -1
        guides[i][1] = -1
        guides[i][2] = 0
        guides[i][3] = -1
        arrivals[i]  = 1
        next_points[i][0] = 0
        next_points[i][1] = 0

    control_n[0] = -1


@jit(nopython=True)
def set_orders(orders, input_orders):
    """
    Update orders by input
    :return: None
    """

    for i in range(10):
        orders[i] = input_orders[i]

### unfinished
def play(time, done, epoch, dev, detect, vision, cars, bullets, bullets_count, orders, acts):
    """
    play game and update display
    :return: None
    """

    render = 1

    # enable keyboard control. render must be '1'
    assert render, 'human play mode, only when render == True'
    while True:
        if not epoch[0] % 10: # get orders every 10 epochs
            if get_keyboard() :
                break
        one_epoch(time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts)


def step(time, done, epoch, render, dev, 
    detect, vision, cars, bullets, bullets_count, orders, acts, 
    screen=None, clock=None, font=None, barriers_rect=None, barriers_img=None, areas_rect=None, 
    areas_img=None, bullet_rect=None, bullet_img=None, chassis_img=None, gimbal_img=None, 
    info_bar_rect=None, info_bar_img=None):
    """
    run game for a single step(10 epoch)
    :param orders: 指令参数
    :return: time, cars, done, detect, vision
    """

    for _ in range(10): # 10 epochs in every step
        one_epoch(time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts)
        if render:
            for event in pygame.event.get(): # get events but not handle
                pass
            update_display(time, dev, detect, vision, cars, bullets, bullets_count, 
                screen, clock, font, barriers_rect, barriers_img, areas_rect, 
                areas_img, bullet_rect, bullet_img, chassis_img, gimbal_img, 
                info_bar_rect, info_bar_img)


@jit(nopython=True)
def one_epoch(time, done, epoch,  detect, vision, cars, bullets, bullets_count, orders, acts):
    """
    make a update for all things
    :return: None
    """

    car_num = 10

    # Move cars in loops
    for n in range(car_num): # loop for every car
        if cars[n, 5] <=0:  # pass cars that were defeated
            cars[n, 5] = 0
            continue
        if not epoch[0] % 10:
            orders_to_acts(orders, acts, n)
        move_car(cars, bullets, bullets_count, acts, vision, n) # Move car
        acts[n, 4] = 0 # disable shoot

    get_camera_vision(cars, vision) # update camera detection

    # Move bullets in loops
    for i in range(car_num):
        bullet_count = bullets_count[i]
        for j in range(bullet_count):
            n = i*100 + j # index of current bullet in bullets array
            if bullets[n]['state'] == 1: # if bullet is still on fly
                move_bullet(cars, bullets, n)

    # Check if game should be ended  
    if time[0] <= 0 \
    or (cars[0:5, 5] == np.zeros(5)).all() \
    or (cars[5:, 5] == np.zeros(5)).all(): 
        # time over or cars of one side are defeated
        done[0] = 1

    if not epoch[0] % 200: # 200 epochs equal to 1s
        time[0] -= 1
    epoch[0] += 1


def update_display(time, dev, detect, vision, cars, bullets, bullets_count, 
        screen, clock, font, barriers_rect, barriers_img, areas_rect, 
        areas_img, bullet_rect, bullet_img, chassis_img, gimbal_img, 
        info_bar_rect, info_bar_img):
    """
    update display
    :return: None
    """

    car_num = 10
    gray    = (180, 180, 180) 
    red     = (190, 20, 20) 
    blue    = (10, 125, 181) 

    screen.fill(gray)
    for i in range(len(barriers_rect)): # render barriers
        screen.blit(barriers_img[i], barriers_rect[i])

    for i in range(len(areas_rect)): # render start areas
        screen.blit(areas_img[i], areas_rect[i])

    for i in range(car_num): # render bullets
        bullet_count = bullets_count[i]
        for j in range(bullet_count):
            n = i*100 + j
            bullet_state = bullets[n]['state']
            if bullet_state != 1:
                continue
            bullet_rect.center = [bullets[n]['x'], bullets[n]['y']]
            screen.blit(bullet_img, bullet_rect)

    for n in range(car_num): # render cars
        chassis_rotate = pygame.transform.rotate(chassis_img, -cars[n, 3] - 90)
        gimbal_rotate = pygame.transform.rotate(gimbal_img, -cars[n, 4] - cars[n, 3] - 90)
        chassis_rotate_rect = chassis_rotate.get_rect()
        gimbal_rotate_rect = gimbal_rotate.get_rect()
        chassis_rotate_rect.center = cars[n, 1:3]
        gimbal_rotate_rect.center = cars[n, 1:3]
        screen.blit(chassis_rotate, chassis_rotate_rect)
        screen.blit(gimbal_rotate, gimbal_rotate_rect)
    # self.screen.blit(self.head_img[0], self.head_rect[0])
    # self.screen.blit(self.head_img[1], self.head_rect[1])

    for n in range(car_num): # render cars info
        select = np.where((vision[n] == 1))[0] + 1
        select2 = np.where((detect[n] == 1))[0] + 1
        info = font.render('{} | {}: {} {}'.format(int(cars[n, 5]), n + 1, select, select2), 
                            True, blue if cars[n, 0] else red)
        screen.blit(info, cars[n, 1:3] + [-20, -60])
        info = font.render('{} {}'.format(int(cars[n, 7]), int(cars[n, 5])), 
                            True, blue if cars[n, 0] else red)
        screen.blit(info, cars[n, 1:3] + [-20, -45])
    info = font.render('time: {}'.format(time[0]), False, (0, 0, 0))
    # self.screen.blit(info, (8, 8))

    if dev: 
        dev_window(cars, screen, font, info_bar_rect, info_bar_img)
    pygame.display.update()
    clock.tick(40)
    pygame.display.set_caption("fps: " + str(clock.get_fps()))


def dev_window(cars, screen, font, info_bar_rect, info_bar_img):
    """
    render dev windows
    :return: None
    """

    car_num = 10
    gray    = (180, 180, 180)
    red     = (190, 20, 20)
    blue    = (10, 125, 181)


    for n in range(car_num): # render outline of cars
        wheels = check_points_wheel(cars[n])
        for w in wheels:
            pygame.draw.circle(screen, blue if cars[n, 0] else red, w.astype(int), 3)
        armors = check_points_armor(cars[n])
        for a in armors:
            pygame.draw.circle(screen, blue if cars[n, 0] else red, a.astype(int), 3)
    screen.blit(info_bar_img, info_bar_rect)

    for n in range(car_num//2): # render cars states
        tags = ['owner', 'x', 'y', 'angle', 'yaw', 'hp', 'can_shoot', 'bullet', 'wheel_hit', 
                'armor_hit', 'car_hit']
        info = font.render('car {}'.format(n), False, (0, 0, 0))
        screen.blit(info, (8 + n * 100, 100))
        for i in range(cars[n].size):
            info = font.render('{}: {}'.format(tags[i], int(cars[n, i])), False, (0, 0, 0))
            screen.blit(info, (8 + n * 100, 117 + i * 17))

    for n in range(car_num//2): # render cars states
        tags = ['owner', 'x', 'y', 'angle', 'yaw', 'hp', 'can_shoot', 'bullet', 'wheel_hit', 
                'armor_hit', 'car_hit']
        info = font.render('car {}'.format(n+5), False, (0, 0, 0))
        screen.blit(info, (8 + n * 100, 350))
        for i in range(cars[n+5].size):
            info = font.render('{}: {}'.format(tags[i], int(cars[n+5, i])), False, (0, 0, 0))
            screen.blit(info, (8 + n * 100, 367 + i * 17))


def get_keyboard(orders, control_n):
    """
    get orders from keyboard
    :return: 0/1
    """

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 1
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_0]: control_n[0] = 0
    if pressed[pygame.K_1]: control_n[0] = 1
    if pressed[pygame.K_2]: control_n[0] = 2
    if pressed[pygame.K_3]: control_n[0] = 3
    if pressed[pygame.K_4]: control_n[0] = 4
    if pressed[pygame.K_5]: control_n[0] = 5
    if pressed[pygame.K_6]: control_n[0] = 6
    if pressed[pygame.K_7]: control_n[0] = 7
    if pressed[pygame.K_8]: control_n[0] = 8
    if pressed[pygame.K_9]: control_n[0] = 9

    n = control_n[0]
    orders[:] = 0

    if pressed[pygame.K_w]: orders[n, 0] += 1
    if pressed[pygame.K_s]: orders[n, 0] -= 1
    if pressed[pygame.K_q]: orders[n, 1] -= 1
    if pressed[pygame.K_e]: orders[n, 1] += 1
    if pressed[pygame.K_a]: orders[n, 2] -= 1
    if pressed[pygame.K_d]: orders[n, 2] += 1
    if pressed[pygame.K_b]: orders[n, 3] -= 1
    if pressed[pygame.K_m]: orders[n, 3] += 1
    if pressed[pygame.K_SPACE]:
        orders[n, 4] = 1
    else:
        orders[n, 4] = 0

    if pressed[pygame.K_n]:
        orders[n, 5] = 1
    else:
        orders[n, 5] = 0

    return 0

    '''
    orders[:] = 0
    # get orders from joysticks
    pygame.joystick.init()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 1

    joystick_count = pygame.joystick.get_count()
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        # axis
        orders[i, 1] -= round(joystick.get_axis(0))  # 左右，q/e，左摇杆
        orders[i, 0] -= round(joystick.get_axis(1))  # 前后，w/s，左摇杆
        orders[i, 2] += round(joystick.get_axis(4))  # 底盘旋转，a/d，LT/RT
        orders[i, 3] -= round(joystick.get_axis(2))  # 云台旋转，b/m，右摇杆

        # button
        orders[i, 4] = joystick.get_button(5)  # 射击，space，RB键
        orders[i, 5] = joystick.get_button(0)  # 自瞄，n，A键

        # hat
        orders[i, 1] += round(joystick.get_hat(0)[0])  # 左右，q/e
        orders[i, 0] += round(joystick.get_hat(0)[1])  # 前后，w/s
        np.round(orders)

        if joystick.get_button(2):
            dev = 1
        else:
            dev = 0

    # return 0
    '''


# useless, pass this func
def get_acts(guides, team, cars_info):
    """
    control car by guides
    :param guides: guides from desicion-making system
    :param team: num of team
    :return: None
    """

    v_max = 1 # max speed

    if len(guides) != 5:
        print('No five paths for all cars!')
        return
    if team == 0:
        car_pos = cars[0:5, 1:3]
        car_rotate = cars[0:5, 3]
    elif team == 1:
        car_pos = cars[5:10, 1:3]
        car_rotate = cars[5:10, 3]
    else:
        print('Team number is wrong!')
        return

    # waypoint tracking, auto aim and shoot
    for i in range(5):
        if arrivals[i] == 1:
            if len(guides[i].path) != 0: # get the first point and delete it from path list
                next_points[i] = guides[i].path[0]
                del guides[i].path[0]
                arrivals[i] = 0

                x_dis = next_points[i][0] - car_pos[i][0]
                y_dis = next_points[i][1] - car_pos[i][1]

                if abs(x_dis)<5 and abs(y_dis)<5: # stop when arrive at target point and update flag
                    acts[i][0] = 0
                    acts[i][1] = 0
                    arrivals[i] = 1
                else:
                    rotate = car_rotate[i]
                    dis = (x_dis**2 + y_dis**2) ** 0.5
                    v_X = v_max * x_dis / dis
                    v_Y = v_max * y_dis / dis
                    # print(i, 'v_X:', v_X, 'v_Y:', v_Y)
                    acts[i][0] = v_X * np.cos(np.deg2rad(rotate)) + v_Y * np.sin(np.deg2rad(rotate))
                    acts[i][1] = v_X * -np.sin(np.deg2rad(rotate)) + v_Y * np.cos(np.deg2rad(rotate))
                    # print(i, 'x_speed:', x_speed, 'y_speed:', y_speed, '\n')
            else:
                print('num', i, 'No path')
        else:
            print('num', i, 'Moving')
            x_dis = next_points[i][0] - car_pos[i][0]
            y_dis = next_points[i][1] - car_pos[i][1]

            if abs(x_dis)<5 and abs(y_dis)<5: # # stop when arrive at target point and update flag
                acts[i][0] = 0
                acts[i][1] = 0
                arrivals[i] = 1
            else:
                rotate = car_rotate[i]
                dis = (x_dis**2 + y_dis**2) ** 0.5
                v_X = v_max * x_dis / dis
                v_Y = v_max * y_dis / dis
                # print(i, 'v_X:', v_X, 'v_Y:', v_Y)
                acts[i][0] = v_X * np.cos(np.deg2rad(rotate)) + v_Y * np.sin(np.deg2rad(rotate))
                acts[i][1] = v_X * -np.sin(np.deg2rad(rotate)) + v_Y * np.cos(np.deg2rad(rotate))
                # print(i, 'x_speed:', x_speed, 'y_speed:', y_speed, '\n')
        print('car num:',i,'next point:',next_points[i])


@jit(nopython=True)
def orders_to_acts(orders, acts, n):
    """
    update acts by orders
    :param orders: control orders
    :param acts: bottom control
    :param n: num of car
    :return: None
    """

    motion        = 6
    rotate_motion = 6
    yaw_motion    = 1

    # x
    acts[n, 0] += orders[n, 0] * 1.5 / motion
    if orders[n, 0] == 0: # when x = 0, x_speed motion
        if acts[n, 0] > 0: acts[n, 0] -= 1.5 / motion
        if acts[n, 0] < 0: acts[n, 0] += 1.5 / motion
    if abs(acts[n, 0]) < 1.5 / motion: acts[n, 0] = 0
    if acts[n, 0] >= 1.5: acts[n, 0] = 1.5
    if acts[n, 0] <= -1.5: acts[n, 0] = -1.5

    # y
    acts[n, 1] += orders[n, 1] * 1 / motion
    if orders[n, 1] == 0: # when y = 0, y_speed motion
        if acts[n, 1] > 0: acts[n, 1] -= 1 / motion
        if acts[n, 1] < 0: acts[n, 1] += 1 / motion
    if abs(acts[n, 1]) < 1 / motion: acts[n, 1] = 0
    if acts[n, 1] >= 1: acts[n, 1] = 1
    if acts[n, 1] <= -1: acts[n, 1] = -1

    # rotate
    acts[n, 2] += orders[n, 2] * 1 / rotate_motion
    if orders[n, 2] == 0: # when rotate = 0, rotate_speed motion
        if acts[n, 2] > 0: acts[n, 2] -= 1 / rotate_motion
        if acts[n, 2] < 0: acts[n, 2] += 1 / rotate_motion
    if abs(acts[n, 2]) < 1 / rotate_motion: acts[n, 2] = 0
    if acts[n, 2] > 1: acts[n, 2] = 1
    if acts[n, 2] < -1: acts[n, 2] = -1

    # yaw
    acts[n, 3] += orders[n, 3] / yaw_motion
    if orders[n, 3] == 0: # when yaw = 0, yaw_speed motion
        if acts[n, 3] > 0: acts[n, 3] -= 1 / yaw_motion
        if acts[n, 3] < 0: acts[n, 3] += 1 / yaw_motion
    if abs(acts[n, 3]) < 1 / yaw_motion: acts[n, 3] = 0
    if acts[n, 3] > 3: acts[n, 3] = 3
    if acts[n, 3] < -3: acts[n, 3] = -3

    acts[n, 4] = orders[n, 4]
    acts[n, 5] = orders[n, 5]

    return orders, acts


@jit(nopython=True)
def move_car(cars, bullets, bullets_count, acts, vision, n):
    """
    Move car
    :param cars: cars states
    :param bullets: bullets states
    :param bullet_count: count of bullet
    :param acts: bottom control
    :param vision: camera detection
    :param n: num of car 0-9
    :return: None
    """

    bullet_speed  = 12.5
    move_discount = 0.6
    theta         = np.rad2deg(np.arctan(45 / 60))

    # move chasis
    if acts[n, 2]:
        p = cars[n, 3]
        cars[n, 3] += acts[n, 2]
        if cars[n, 3] > 180: cars[n, 3] -= 360
        if cars[n, 3] < -180: cars[n, 3] += 360
        if check_interface(cars, n):
            acts[n, 2] = -acts[n, 2] * move_discount
            cars[n, 3] = p

    # move gimbal
    if acts[n, 3]:
        cars[n, 4] += acts[n, 3]
        if cars[n, 4] > 90: cars[n, 4] = 90
        if cars[n, 4] < -90: cars[n, 4] = -90

    # auto aim
    if acts[n, 5]:
        select = np.where((vision[n] == 1))[0]
        if select.size:
            angles = np.zeros(select.size)
            for ii, i in enumerate(select):
                if cars[n, 0] == cars[i, 0]: # pass teammates
                    continue
                x, y = cars[i, 1:3] - cars[n, 1:3]
                angle = np.angle(x + y * 1j, deg=True) - cars[i, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if angle >= -theta and angle < theta:
                    armor = get_armor(cars[i], 2)
                elif angle >= theta and angle < 180 - theta:
                    armor = get_armor(cars[i], 3)
                elif angle >= -180 + theta and angle < -theta:
                    armor = get_armor(cars[i], 1)
                else:
                    armor = get_armor(cars[i], 0)
                x, y = armor - cars[n, 1:3]
                angle = np.angle(x + y * 1j, deg=True) - cars[n, 4] - cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angles[ii] = angle
            m = np.where(np.abs(angles) == np.abs(angles).min())
            cars[n, 4] += angles[m][0]
            if cars[n, 4] > 90: cars[n, 4] = 90
            if cars[n, 4] < -90: cars[n, 4] = -90

    # move position
    if acts[n, 0] or acts[n, 1]:
        angle = np.deg2rad(cars[n, 3])
        # x
        p = cars[n, 1]
        cars[n, 1] += (acts[n, 0]) * np.cos(angle) - (acts[n, 1]) * np.sin(angle)
        if check_interface(cars, n):
            acts[n, 0] = -acts[n, 0] * move_discount
            cars[n, 1] = p
        # y
        p = cars[n, 2]
        cars[n, 2] += (acts[n, 0]) * np.sin(angle) + (acts[n, 1]) * np.cos(angle)
        if check_interface(cars, n):
            acts[n, 1] = -acts[n, 1] * move_discount
            cars[n, 2] = p

    # shoot
    if acts[n, 4] and cars[n, 7]:
        if cars[n, 6]:
            cars[n, 7] -= 1
            bullets_count[n] += 1
            bullets[n*100+bullets_count[n]]['x'] = cars[n, 1]
            bullets[n*100+bullets_count[n]]['y'] = cars[n, 2]
            bullets[n*100+bullets_count[n]]['angle'] = cars[n, 3] + cars[n, 4]
            bullets[n*100+bullets_count[n]]['speed'] = bullet_speed
            bullets[n*100+bullets_count[n]]['owner'] = n
            bullets[n*100+bullets_count[n]]['state'] = 1

            cars[n, 6] = 0
        else:
            cars[n, 6] = 1
    else:
        cars[n, 6] = 1

    return cars, bullets, acts


@jit(nopython=True)
def move_bullet(cars, bullets, n):
    """
    move bullets and check hit/miss
    :param bullets: bullets states
    :param n: index of bullet 0-999
    return Bool
    """

    car_num    = 10
    map_length = 800
    map_width  = 600

    old_x = bullets[n]['x']
    old_y = bullets[n]['y']
    old_point = np.array([old_x, old_y], dtype=np.float32)
    cur_x = old_x + bullets[n]['speed'] * np.cos(np.deg2rad(bullets[n]['angle']))
    cur_y = old_y + bullets[n]['speed'] * np.sin(np.deg2rad(bullets[n]['angle']))
    bullets[n]['x'] = cur_x
    bullets[n]['y'] = cur_y
    cur_point = np.array([cur_x, cur_y], dtype=np.float32)
    line_points = np.array([[old_x, old_y],
                             [cur_x, cur_y]], dtype=np.float32)

    # out of border
    if bullets[n]['x'] <= 0 or bullets[n]['x'] >= map_length \
            or bullets[n]['y'] <= 0 or bullets[n]['y'] >= map_width: 
        bullets[n]['state'] = 2

    # hit barrier
    if line_barriers_check(cur_point, old_point): 
        bullets[n]['state'] = 2

    # hit armor
    for i in range(car_num):
        if i == bullets[n]['owner']: 
            continue
        if np.abs(cur_point - cars[i, 1:3]).sum() < 52.5:
            points = transfer_to_car_coordinate(cars[i], line_points)
            if segment(points[0], points[1], [-18.5, -5], [-18.5, 6]) \
            or segment(points[0], points[1], [18.5, -5], [18.5, 6]) \
            or segment(points[0], points[1], [-5, 30], [5, 30]) \
            or segment(points[0], points[1], [-5, -30], [5, -30]):
                cars[i, 5] -= 50 # HP-50 for every bullet hit
                bullets[n]['state'] = i+10
            if line_rect_check(points[0], points[1], [-18, -29, 18, 29]): 
                bullets[n]['state'] = 2
    
    return bullets


@jit(nopython=True)
def cross(p1, p2, p3):
    """
    calculate cross product
    :param p1: position 1
    :param p2: position 2
    :param p3: position 3
    :return x1 * y2 - x2 * y1
    """

    # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    
    return x1 * y2 - x2 * y1


@jit(nopython=True)
def segment(p1, p2, p3, p4):
    """
    if two segments intersect with each other
    :param p1: starting point of seg a
    :param p2: endpoint of seg a
    :param p3: starting point of seg b
    :param p4: endpoint 2 of seg b
    :return: 0/1
    """

    # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
    if (max(p1[0], p2[0]) >= min(p3[0], p4[0]) \
    and max(p3[0], p4[0]) >= min(p1[0], p2[0]) \
    and max(p1[1], p2[1]) >= min(p3[1], p4[1]) \
    and max(p3[1], p4[1]) >= min(p1[1], p2[1])): # if rects of segs intersect
        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0 \
        and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0): # if cross 
            return 1
        else:
            return 0
    else:
        return 0


@jit(nopython=True)
def line_rect_check(l1, l2, re):
    """
    if segment intersects with rectangle 
    :param l1: starting point of seg a
    :param l2: endpoint of seg a
    :param re: four points of rect
    :return: 0/1
    """
    
    # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
    # check if line cross rect, re = [x_leftdown, y_leftdown, x_rightup, y_rightup]
    p1 = [re[0], re[1]]
    p2 = [re[2], re[3]]
    p3 = [re[2], re[1]]
    p4 = [re[0], re[3]]
    if segment(l1, l2, p1, p2) or segment(l1, l2, p3, p4):
        return 1
    else:
        return 0


@jit(nopython=True)
def line_barriers_check(l1, l2):
    """
    if segment intersects with barriers
    :param barriers: barriers rects
    :param l1: starting point of seg a
    :param l2: endpoint of seg a
    :return: 0/1
    """

    barriers = np.array([[350.0, 450.0, 250.0, 275.0], # O1
                    [350.0, 450.0, 275.0, 300.0], # O1 -
                    [350.0, 450.0, 300.0, 325.0], # O1 -
                    [350.0, 450.0, 325.0, 350.0], # O1 -
                    [200.0, 300.0,  87.0, 113.0], # O2 -
                    [200.0, 300.0, 487.0, 513.0], # O3 -
                    [500.0, 600.0,  87.0, 113.0], # O4 -
                    [500.0, 600.0, 487.0, 513.0], # O5 -
                    [187.0, 213.0, 100.0, 200.0], # O2 |
                    [187.0, 213.0, 400.0, 500.0], # O3 |
                    [587.0, 613.0, 100.0, 200.0], # O4 |
                    [587.0, 613.0, 400.0, 500.0], # O5 | 
                    ], dtype=np.float32) # barriers float32-ax4

    for b in barriers:
        re = [b[0], b[2], b[1], b[3]]  # [left, up, right, down]
        if line_rect_check(l1, l2, re):  
            return 1
    return 0


@jit(nopython=True)
def line_cars_check(cars, l1, l2): 
    """
    if segment intersects with cars
    :param cars: cars states 
    :param l1: starting point of seg a
    :param l2: endpoint of seg a
    :return: 0/1
    """

    for car in cars:
        if (car[1:3] == l1).all() or (car[1:3] == l2).all():
            continue
        p1, p2, p3, p4 = get_car_outline(car)
        if segment(l1, l2, p1, p2) or segment(l1, l2, p3, p4): 
            return 1
    return 0


@jit(nopython=True)
def get_lidar_vision(cars, detect):
    """
    update lidar detection
    :param cars: cars states
    :param detect: lidar detection matrix
    :return: detect
    """

    lidar_angle = 120 / 2
    car_num     = 10

    for n in range(car_num):
        for i in range(car_num - 1):
            if cars[n, 0] == cars[n-i-1, 0]:
                continue
            x, y = cars[n - i - 1, 1:3] - cars[n, 1:3]
            angle = np.angle(x + y * 1j, deg=True)
            if angle >= 180: angle -= 360
            if angle <= -180: angle += 360
            angle = angle - cars[n, 3]
            if angle >= 180: angle -= 360
            if angle <= -180: angle += 360
            if abs(angle) < lidar_angle: # if within lidar angle 
                if line_barriers_check(cars[n, 1:3], cars[n - i - 1, 1:3]) \
                        or line_cars_check(cars, cars[n, 1:3], cars[n - i - 1, 1:3]):
                    detect[n, n - i - 1] = 0
                else:
                    detect[n, n - i - 1] = 1
            else:
                detect[n, n - i - 1] = 0

    return detect


@jit(nopython=True)
def get_camera_vision(cars, vision):
    """
    update camera detection
    :param cars: cars states
    :param detect: camera detection matrix
    :return: vision
    """

    camera_angle = 75 / 2 
    car_num      = 10

    for n in range(car_num):
        for i in range(car_num - 1):
            if cars[n, 0] == cars[n-i-1, 0]:
                continue
            x, y = cars[n - i - 1, 1:3] - cars[n, 1:3]
            angle = np.angle(x + y * 1j, deg=True)
            if angle >= 180: angle -= 360
            if angle <= -180: angle += 360
            angle = angle - cars[n, 4] - cars[n, 3]
            if angle >= 180: angle -= 360
            if angle <= -180: angle += 360
            if abs(angle) < camera_angle: # if within camera angle 
                if line_barriers_check(cars[n, 1:3], cars[n - i - 1, 1:3]) \
                        or line_cars_check(cars, cars[n, 1:3], cars[n - i - 1, 1:3]): 
                    vision[n, n - i - 1] = 0
                else:
                    vision[n, n - i - 1] = 1
            else:
                vision[n, n - i - 1] = 0

    return vision


@jit(nopython=True)
def transfer_to_car_coordinate(car, points):
    """
    transfer from global coordinate to car coordinate
    :param car: car state
    :param points: points to be transfered
    :param n: num of car 0-9
    :return: new_points @ rotate_matrix
    """

    pan_vecter = -car[1:3]
    rotate_matrix = np.array([[np.cos(np.deg2rad(car[3] + 90)), -np.sin(np.deg2rad(car[3] + 90))],
                                [np.sin(np.deg2rad(car[3] + 90)), np.cos(np.deg2rad(car[3] + 90))]], 
                                dtype=np.float32)
    new_points = points + pan_vecter 

    return new_points @ rotate_matrix


@jit(nopython=True)
def check_points_wheel(car):
    """
    get wheels position in global coordinate
    :param car: car state
    :return: xs @ rotate_matrix + car[1:3]
    """

    rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]], 
                                dtype=np.float32)
    xs = np.array([[-22.5, -29], [22.5, -29],
                    [-22.5, -14], [22.5, -14],
                    [-22.5, 14], [22.5, 14],
                    [-22.5, 29], [22.5, 29]], dtype=np.float32)
    return xs @ rotate_matrix + car[1:3] 


@jit(nopython=True)
def check_points_armor(car):
    """
    get armors position in global coordinate
    :param car: car state
    :return: xs @ rotate_matrix + car[1:3] 
    """

    rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]], 
                                dtype=np.float32)
    xs = np.array([[-6.5, -30], [6.5, -30],
                    [-18.5, -7], [18.5, -7],
                    [-18.5, 0], [18.5, 0],
                    [-18.5, 6], [18.5, 6],
                    [-6.5, 30], [6.5, 30]], dtype=np.float32)
    return xs @ rotate_matrix + car[1:3] 


@jit(nopython=True)
def get_car_outline(car):
    """
    get car outline position in global coordinate
    :param car: car state
    :return: xs @ rotate_matrix + car[1:3]
    """

    rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]], 
                                dtype=np.float32)
    xs = np.array([[-22.5, -30], 
                    [22.5, 30], 
                    [-22.5, 30], 
                    [22.5, -30]], dtype=np.float32)

    return xs @ rotate_matrix + car[1:3]


@jit(nopython=True)
def get_armor(car, i):
    """
    get armor i position in global coordinate
    :param car: car state 
    :param: i: num of armor 0-3
    :return: x @ rotate_matrix + car[1:3]
    """

    rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3] + 90)), -np.sin(-np.deg2rad(car[3] + 90))],
                                [np.sin(-np.deg2rad(car[3] + 90)), np.cos(-np.deg2rad(car[3] + 90))]], dtype=np.float32)
    xs = np.array([[0.0, -30], 
                    [18.5, 0.0], 
                    [0.0, 30], 
                    [-18.5, 0.0]], dtype=np.float32)
    x = xs[i]

    return  x @ rotate_matrix + car[1:3]


@jit(nopython=True)
def check_interface(cars, n):
    """
    check if car hit border/barrier/car
    :param cars: cars states
    :param n: num of car
    :return: 0/1
    """

    map_length = 800
    map_width  = 600
    car_num    = 10

    barriers = np.array([[350.0, 450.0, 250.0, 275.0], # O1
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
                    ], dtype=np.float32) # barriers float32-ax4

    # wheels hit border/barrier
    wheels = check_points_wheel(cars[n])
    for w in wheels: 
        if w[0] <= 0 or w[0] >= map_length or w[1] <= 0 or w[1] >= map_width:
            cars[n, 8] += 1
            return 1
        for b in barriers:
            if w[0] >= b[0] and w[0] <= b[1] and w[1] >= b[2] and w[1] <= b[3]:
                cars[n, 8] += 1
                return 1

    # armor hit border/barrier
    armors = check_points_armor(cars[n])
    for a in armors: 
        if a[0] <= 0 or a[0] >= map_length or a[1] <= 0 or a[1] >= map_width:
            cars[n, 9] += 1
            cars[n, 5] -= 10
            return 1
        for b in barriers:
            if a[0] >= b[0] and a[0] <= b[1] and a[1] >= b[2] and a[1] <= b[3]:
                cars[n, 10] += 1
                cars[n, 5] -= 10
                return 1

    # car hit car
    for i in range(car_num):
        if i == n: 
            continue
        wheels_tran = transfer_to_car_coordinate(cars[i], wheels)
        for w in wheels_tran:
            if w[0] >= -22.5 and w[0] <= 22.5 and w[1] >= -30 and w[1] <= 30:
                cars[n, 10] += 1
                return 1
        armors_tran = transfer_to_car_coordinate(cars[i], armors)
        for a in armors_tran:
            if a[0] >= -22.5 and a[0] <= 22.5 and a[1] >= -30 and a[1] <= 30:
                cars[n, 10] += 1
                cars[n, 5] -= 10
                return 1
    return 0


# test function below
def runtest(render=0, dev=0):
    # constant
    map_length = 800
    map_width  = 600
    barriers   = np.array([[350.0, 450.0, 250.0, 275.0], # O1 -
                            [350.0, 450.0, 275.0, 300.0], # O1 -
                            [350.0, 450.0, 300.0, 325.0], # O1 -
                            [350.0, 450.0, 325.0, 350.0], # O1 -
                            [200.0, 300.0,  87.0, 113.0], # O2 -
                            [200.0, 300.0, 487.0, 513.0], # O3 -
                            [500.0, 600.0,  87.0, 113.0], # O4 -
                            [500.0, 600.0, 487.0, 513.0], # O5 -
                            [187.0, 213.0, 100.0, 200.0], # O2 |
                            [187.0, 213.0, 400.0, 500.0], # O3 |
                            [587.0, 613.0, 100.0, 200.0], # O4 |
                            [587.0, 613.0, 400.0, 500.0], # O5 | 
                            ], dtype=np.float32) # barriers
    areas      = np.array([[[0.0, 100.0, 0.0, 100.0],
                            [0.0, 100.0, 125.0, 225.0],
                            [0.0, 100.0, 250.0, 350.0],
                            [0.0, 100.0, 375.0, 475.0],
                            [0.0, 100.0, 500.0, 600.0]],
                            [[700.0, 800.0, 0.0, 100.0],
                            [700.0, 800.0, 125.0, 225.0],
                            [700.0, 800.0, 250.0, 350.0],
                            [700.0, 800.0, 375.0, 475.0],
                            [700.0, 800.0, 500.0, 600.0]]], dtype=np.float32) # start areas

    input_orders = np.ones((10, 6), dtype=np.int8)

    # initialize
    start = timeit.default_timer()
    time, done, epoch, detect, vision, cars, bullets, bullets_count, \
    fake_orders, acts, guides, arrivals, next_points, control_n \
    = init()
    reset(time, done, epoch, detect, vision, cars, bullets, bullets_count, \
            fake_orders, acts, guides, arrivals, next_points, control_n)
    end = timeit.default_timer()
    print("Elapsed (with compilation) init() x 1 + reset() x 1 = %s" % (end - start))

    if render:
        pygame.init()
        screen = pygame.display.set_mode((map_length, map_width))
        pygame.display.set_caption('RoboSim')

        # load barrier img
        barriers_img = []
        barriers_rect = []
        for i in range(barriers.shape[0]):
            barriers_img.append(pygame.image.load('./imgs/barrier_{}.png'.format('horizontal' 
                                                    if i < 8 else 'vertical')))
            barriers_rect.append(barriers_img[-1].get_rect())
            barriers_rect[-1].center = [barriers[i][0:2].mean(), barriers[i][2:4].mean()]

        # load start area and car
        areas_img = []
        areas_rect = []
        for oi, o in enumerate(['red', 'blue']):
            for ti in range(5): # start area
                areas_img.append(pygame.image.load('./imgs/area_start_{}.png'.format(o)))
                areas_rect.append(areas_img[-1].get_rect())
                areas_rect[-1].center = [areas[oi, ti][0:2].mean(), areas[oi, ti][2:4].mean()]

        chassis_img = pygame.image.load('./imgs/chassis_g.png') # chasis
        gimbal_img = pygame.image.load('./imgs/gimbal_g.png') # gimbal
        bullet_img = pygame.image.load('./imgs/bullet_s.png') # bullet
        bullet_rect = bullet_img.get_rect()
        info_bar_img = pygame.image.load('./imgs/info_bar.png') # info bar
        info_bar_rect = info_bar_img.get_rect()
        info_bar_rect.center = [200, map_width / 2]

        pygame.font.init()
        font = pygame.font.SysFont('info', 20)
        clock = pygame.time.Clock()

        # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
        start = timeit.default_timer()
        set_orders(fake_orders, input_orders)
        step(time, done, epoch, render, dev, 
            detect, vision, cars, bullets, bullets_count, fake_orders, acts, 
            screen, clock, font, barriers_rect, barriers_img, areas_rect, 
            areas_img, bullet_rect, bullet_img, chassis_img, gimbal_img, 
            info_bar_rect, info_bar_img)
        set_orders(fake_orders, input_orders)
        end = timeit.default_timer()
        print("Elapsed (with compilation) s_o() x 2 + step() x 1 = %s" % (end - start))

        # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
        start = timeit.default_timer()
        for i_episode in range(10):
            for j_step in range(10):
                step(time, done, epoch, render, dev, 
                    detect, vision, cars, bullets, bullets_count, fake_orders, acts, 
                    screen, clock, font, barriers_rect, barriers_img, areas_rect, 
                    areas_img, bullet_rect, bullet_img, chassis_img, gimbal_img, 
                    info_bar_rect, info_bar_img)
                set_orders(fake_orders, input_orders)
                # print('epoch- ', epoch,  'bullets_count: ', bullets_count)
            reset(time, done, epoch, detect, vision, cars, bullets, bullets_count, 
            fake_orders, acts, guides, arrivals, next_points, control_n)
            set_orders(fake_orders, input_orders)
        end = timeit.default_timer()
        print("Elapsed (after compilation) s_o() x 110 + step() x 100 + reset() x 10 = %s" % (end - start))

    else:
        # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
        start = timeit.default_timer()
        set_orders(fake_orders, input_orders)
        step(time, done, epoch, render, dev, 
            detect, vision, cars, bullets, bullets_count, fake_orders, acts)
        set_orders(fake_orders, input_orders)
        end = timeit.default_timer()
        print("Elapsed (with compilation) s_o() x 2 + step() x 1 = %s" % (end - start))

        # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
        start = timeit.default_timer()
        for i_episode in range(10):
            for j_step in range(10):
                step(time, done, epoch, render, dev, 
                    detect, vision, cars, bullets, bullets_count, fake_orders, acts)
                set_orders(fake_orders, input_orders)
                # print('epoch- ', epoch,  'bullets_count: ', bullets_count)
            reset(time, done, epoch, detect, vision, cars, bullets, bullets_count, \
            fake_orders, acts, guides, arrivals, next_points, control_n)
            set_orders(fake_orders, input_orders)
        end = timeit.default_timer()
        print("Elapsed (after compilation) s_o() x 110 + step() x 100 + reset() x 10 = %s" % (end - start))
    

def runfunctest():
    pass


if __name__ == "__main__": 
    # cProfile.run('runtest(0,0)', sort='tottime')
    # t1 = Timer("testv()", "from __main__ import testv")
    # print("transfer_to_car_coordinate: ",t1.timeit(number=1000), "seconds")

    runtest(0,0)
    
    
    