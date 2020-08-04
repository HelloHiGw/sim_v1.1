# SimInterface

## 一、设计观测值与奖励函数

首先根据自己的需要完成封装类`SimInterface`中的`get_observation`和`get_reward`部分，即下面两部分

```python
    def get_observation(self):
        # personalize your observation here
        pass
    def get_reward(self):
        # personalize your reward here
        pass
```


## 二、开始使用

`SimInterface`的使用方式与[openai](https://openai.com/)的[gym](https://github.com/openai/gym)类似

### 1、初始化

导入，声明并初始化，`render`为`1`时显示会画面.

```python
from SimInterface import SimInterface
sim = SimInterface(time=car_num, render=1, dev=1)
sim.reset()
```

### 2、执行一步

传入决策，得到观测，奖励，是否结束，和其他信息，参数的具体格式请参考：[params.md](./params.md)

```python
# action format (int8, np.array): [['x', 'y', 'rotate', 'yaw', 'shoot', 'auto_aim'], ...]
# action.shape = (car_num, 6)
actions = np.ones((10,6), dtype=numpy.int8)
_ = sim.step(actions)
```

### 3、使用键盘控制

暂不支持
