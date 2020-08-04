### 相关模块
point.py `坐标点Point-类`
gridmap.py `静态栅格地图（包含障碍物）,地图坐标从(1,1)开始`
convertc.py `实际坐标和栅格地图坐标转换`
a_star.py `给定栅格地图中的起点坐标和终点坐标进行路径规划`

### 使用说明
1 已知实际地图中起点坐标`(X_start, Y_start)`和终点坐标`(X_end, Y_end)`,假设坐标系以厘米为单位，指车辆定位坐标(一般为车辆中心);
2 将实际地图坐标系中的坐标转化为栅格地图坐标系中所属栅格块的坐标`(x_start, y_start)`和`(x_end, y_end)`;
```
    StartPoint = point.Point(X_start, Y_start)
    EndPoint   = point.Point(X_end, Y_end)
    startPoint = convertc.real2grid([StartPoint], ratio)[0] # ratio为栅格对应的实际长度
    endPoint   = convertc.real2grid([EndPoint], ratio)[0]
```
3 调用AStar算法进行路径规划，获得栅格坐标系中的路径点`Point类`列表`path`
```
    astar = a_star.AStar(startPoint, endPoint, ratio)
    path = astar.GetPath()
```
4 将栅格坐标系路径点列表转化为实际坐标系路径点列表`path_real`
```
    path_real = convertc.grid2real(path)
```

注: 运行示例见a_star.py中main函数