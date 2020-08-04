import point

def real2grid(rpoint_list, ratio):
    """
    将实际坐标系坐标点转化为栅格坐标系坐标点
    :param rpoint_list: 实际坐标系坐标点列表 
    :param ratio: 栅格对应的实际长度
    :return gpoint_list: 栅格坐标系坐标点列表
    """

    gpoint_list = []

    for rp in rpoint_list:
        gp_x = rp.x//ratio + 1
        gp_y = rp.y//ratio + 1
        gpoint_list.append(point.Point(gp_x, gp_y))

    return gpoint_list


def grid2real(gpoint_list, ratio):
    """
    将栅格坐标系坐标点(栅格中心)转化为实际坐标系坐标点
    :param gpoint_list: 栅格坐标系坐标点列表
    :param ratio: 栅格对应的实际长度
    :return rpoint_list: 实际坐标系坐标点列表
    """

    rpoint_list = []

    for gp in gpoint_list:
        rp_x = gp.x * ratio - ratio//2
        rp_y = gp.y * ratio - ratio//2
        rpoint_list.append([rp_x, rp_y])

    return rpoint_list