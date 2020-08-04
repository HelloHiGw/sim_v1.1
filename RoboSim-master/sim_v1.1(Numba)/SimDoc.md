# RoboSim使用指南

## 一、须知事项

### 环境依赖
见[README.md](../README.md)

### 说明文档
参数格式说明：[params.md](./demo/docs/params.md)
kernal.py说明：[kernal.md](./demo/docs/kernal.md)
SimInterface.py说明：[SimInterface.md](./demo/docs/SimInterface.md)

### 注意事项
请勿使用python list等可变对象, 使用numpy array进行代替
变量类型尽量使用float32, int32, int8等，以免Numba不兼容导致报错

## 二、结构介绍

`kernal`的核心函数是`one_epoch`，表示运行一个周期，一个周期里会调用：`move_car`，`move_bullet`，同时会更新摄像头视野信息等，并更新游戏画面（如果开启渲染）

可以调用`one_epoch`的有两个函数：`step`和`play`(暂未完成)。`step`做的是获取传入的指令`orders`，然后运行10个周期；`play`的唯一区别在于：它会一直运行，然后每十个周期会获得一次`orders`。

## 三、运行测试

### 1、运行速度

以下为当前不开启可视化的测试结果：
*注*
1、目前Python代码已使用Numba加速，使用前请安装NumPy和Numba，具体版本见readme.md，优化前指未使用Numba加速时。

|函数名|编译运行时间/s|优化后(10000 loops)/ms|优化前(10000 loops)/ms|Speedup|
|-|-|-|-|-|
|empty 10000 loops|||0.978/0.558|-|
|one_epoch without @jit|17.812330|6.512753s(100 loops)|55.259s(100 loops)|8.49|
|all @jit|26.9828574|1.847460s(100 loops)|55.259s(100 loops)|30.06|
|transfer_to_car_coordinate|1.037945|35.567|337.665|9.49|
|segment|0.523856|14.762|176.354|11.95|
|cross|0.297839|8.458|21.935|2.59|

测试环境为：Windows 10，2核i5-7200U CPU 2.71GHz，不开渲染的情况下100step(1000 epoch)所花时间在1.8s左右。

测试代码：sim_test.py

#### Numba加速效果说明
Numba加速效果与硬件水平有关，每次加速后代码运行时间会有一些差异，即多次加速后运行，所花时间方差较大。

### 2、并行比赛

利用`SimInterface.py`可以同时进行多场比赛，可以提高学习速度

### 3、联机对抗

（暂未完成）
实现人与人对抗的目的在于可以进行模仿学习，模仿学习的想法来源于[DeepMind](https://deepmind.com/)的[AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/)

#### 操作指南

其他的部分基本不用动，改变获取指令的方法，改为联网获取，另外还可以在`get_order`函数里将云台的控制方式改为用鼠标控制

### 4、增加随机误差

模拟器毕竟不是真实世界，增加一些随机性有助于提高模拟到实际迁移能力，想法来源于[OpenAI](https://openai.com/)的研究[Generalizing from Simulation](https://blog.openai.com/generalizing-from-simulation/)

#### 操作指南

在函数`move_car`的开头，对`self.acts`增加一些误差，关于`acts`的具体细节，可参见[params.md](./params.md)，注意是`kernel`里的`acts`

### 5、视野

激光雷达和摄像头的视野用来表示能检测到车，当某个车在摄像头视野内时，可以自动瞄准这个车。现在使用的视野算法为：首先检测角度是不是符合，再检查两车中心联线上是否有阻碍（障碍物或车）。这样做的问题是：在有些刁钻的角度，会出现不合理的视野

## 四、仿真环境开发手册-Numba加速
### 1 变量
#### 1.1 常量
|名称|类型|含义|
|-|-|-|
|map_length|int32|地图长度|
|map_width|int32|地图宽度|
|car_num|int32|车辆数量|
|render|int8|是否渲染|
|bullet_speed|float32|子弹速度|
|motion|float32|移动惯性|
|rotate_motion|float32|底盘旋转的惯性|
|yaw_motion|float32|云台旋转的惯性|
|camera_angle|float32|摄像头的视野范围|
|lidar_angle|float32|激光雷达的视野范围|
|move_discount|float32|撞墙后反弹的强度|
|theta|float32|自瞄参考角|
|areas|float32-2x5x4|起始区域|
|barriers|float32-ax4|障碍物区域|

*注:* 
barrier = [x1, x2, y1, y2]

#### 1.2 比赛状态量
|名称|类型|含义|
|-|-|-|
|time|int32-1|比赛时间|
|done|int8-1|是否结束|
|epoch|int32-1|更新轮数|
|dev|int8-1|是否进入开发者模式|
|detect|int8-10x10|雷达探测结果|
|vision|int8-10x10|视觉探测结果|
|cars|float32-10x11|所有车辆状态信息|
|bullets|float32/int32-nx6|所有子弹状态信息|
|bullets_count|int32-10|已使用子弹计数|

*注:* 
car = [team, x, y, rotate, yaw, HP, can_shoot, bullet_num, wheel_hit, armor_hit, car_hit]
bullet = [x, y, angle, speed, owner, state]

#### 1.3 中间变量
|名称|类型|含义|
|-|-|-|
|orders|int8-10x6|控制指令|
|acts|float32-10x6|动作信息|
|control_n|int32-1|键盘控制的车辆对应编号|
|guides|int32-5x4|决策指示信息|
|arrivals|int8-5|是否到达下一个路径点|
|next_points|int32-5x2|下一组路径点|

*注:* 
order = [x, y, rotate, yaw, shoot, auto_aim]
act = [x_speed, y_speed, rotate_speed, yaw_speed, shoot, auto_aim]
guide = [x, y, shoot, target_num]
next_point = [x, y]

### 2 函数
#### 2.1 初始化函数
|名称|参数类型|返回值|含义|优化|
|-|-|-|-|-|
|init|None|time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts, guides, arrivals, next_points, control_n|初始化所有变量|jit|
||None|cars, bullets, bullets_count|重置车辆和子弹|jit|
|reset|None|orders, acts, guides, arrivals, next_points, control_n|重置中间变量|jit|

#### 2.2 运行函数
|名称|参数类型|返回值|含义|
|-|-|-|-|
|play(cars,guides,endFlag)|float32-10x11, int32-5x4, int8|None|持续运行仿真|
|step(time, done, epoch, render, dev, detect, vision, cars, bullets, bullets_count, orders, acts, screen, clock, font, barriers_rect, barriers_img, areas_rect, areas_img, bullet_rect, bullet_img, chassis_img, gimbal_img, info_bar_rect, info_bar_img)|int8-10x6|time, done, detect, vision, cars|单步运行仿真|
|reset|time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts, guides, arrivals, next_points, control_n|None|重置所有变量|
|one_epoch(time, done, epoch, detect, vision, cars, bullets, bullets_count, orders, acts)|None|None|更新一轮仿真|
|update_display(time, dev, detect, vision, cars, bullets, bullets_count, screen, clock, font, barriers_rect, barriers_img, areas_rect, areas_img, bullet_rect, bullet_img, chassis_img, gimbal_img, info_bar_rect, info_bar_img)|None|None|更新渲染画面|
|dev_window(cars, screen, font, info_bar_rect, info_bar_img)|None|None|开发者窗口|
|get_keyboard(orders, control_n)|float32-10x6 int32-1|int8|获取键盘指令|
|set_orders(orders, input_orders)|float32-10x6 int32-1|int8|根据输入更新指令|

#### 2.3 比赛功能函数
|名称|参数类型|返回值|含义|进度|优化|
|-|-|-|-|-|-|
|orders_to_acts(orders, acts, n)|float32-10x11, int32-10x6, float32-10x6|float32-10x6 float32-10x6|指令转化为动作|jit|
|move_car(cars, bullets, bullet_count, acts, vision, n)|int32|None|移动底盘、云台和位置及开火|jit|
|move_bullet(cars, bullets, n)|int32|None|移动子弹及碰撞检测|jit|
|cross(p1,p2,p3)|float32-2, float32-2, float32-2|float32|叉积计算|jit|
|segment(p1,p2,p3,p4)|float32-2, float32-2, float32-2, float32-2|int8|线段与线段相交判断|jit|
|line_rect_check(l1,l2,re)|float32-2, float32-2, float32-4|int8|线段与矩形相交判断|jit|
|line_barriers_check(l1, l2)|float32-2, float32-2|int8|线段与障碍物相交判断|jit|
|line_cars_check(cars, l1, l2)|float32-2, float32-2|int8|线段与车辆相交判断|jit|
|get_lidar_vision(cars, detect)|None|int8-10x10|计算雷达探测结果|jit|
|get_camera_vision(cars, vison)|None|int8-10x10|计算视觉探测结果|jit|
|transfer_to_car_coordinate(car, points)|float32-2x2, int32|float32-2x2|全局坐标系到车辆坐标系的转换|jit|
|check_points_wheel(car)|float32-11|float32-8x2|获取车轮全局位置|jit|
|check_points_armor(car)|float32-11|float32-8x2|获取所有装甲板全局位置|jit|
|get_car_outline(car)|float32-11|float32-4x2|获取车辆轮廓全局位置|jit|
|get_armor(car,i)|float32-11, int32|float32-2|获取装甲i位置信息|jit|
|check_interface(cars, n)|int32|int8|检查车辆碰撞情况|jit|

