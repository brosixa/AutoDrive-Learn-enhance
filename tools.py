import cv2
import numpy as np
import math

# 特征提取，只需要某一象限的视图
def ROI(edge, region):
    h, w = edge.shape
    mask = np.zeros_like(edge)

    if region == 1:  # 左上
        polygon = np.array([[(0, 0),
                             (0, h / 2),
                             (w / 2, 0),
                             (w / 2, h / 2)]], np.int32)

    elif region == 2:  # 右上
        polygon = np.array([[(w, 0),
                             (w, h / 2),
                             (w / 2, h / 2),
                             (w / 2, 0)]], np.int32)

    elif region == 3:  # 左下
        polygon = np.array([[(0, h),
                             (0, h / 2),
                             (w / 2, h / 2),
                             (w / 2, h)]], np.int32)
    else:  # 右下
        polygon = np.array([[(w, h),
                             (w, h / 2),
                             (w / 2, h / 2),
                             (w / 2, h)]], np.int32)

    cv2.fillPoly(mask, polygon, 255)  # 将mask中的多边形区域polygon涂白

    roi_edge = cv2.bitwise_and(edge, mask)  # 将edge和mask进行按位与运算，只保留白色部分
    return roi_edge


# 获取直线y = k * x + b在frame上的端点
def get_line(h, w, fit):
    k, b = fit
    y1 = h
    y2 = int(y1 / 2)
    x1 = max(-w, min(2 * w, int((y1 - b) / k)))
    x2 = max(-w, min(2 * w, int((y2 - b) / k)))
    return [[x1, y1, x2, y2]]


def detect_line(edge, direction):
    lane = []
    # 基于霍夫变换进行直线检测
    # 精度为1像素，角度精度1度， 信任阈值10， 最短长度8，最短间隙8
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 10, np.array([]), 8, 8)

    if lines is None:
        print('没有检测到线段')
        return []
    h, w = edge.shape
    fits = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            k = fit[0]  # 斜率
            b = fit[1]  # 截距
            if direction == 'left' and k < 0:
                fits.append((k, b))
                # print('left')
            elif direction == 'right' and k > 0:
                fits.append((k, b))
                # print('right')
    # print(fits)
    if len(fits) > 0:
        fit_average = np.average(fits, axis=0)  # axis=0，分别对斜率和截距求平均
        lane.append(get_line(h, w, fit_average))
        return lane
    return []



def get_action(h, w, yellow_lane, white_lane, pid):
    mid = w / 2
    x_offset = 0

    if yellow_lane and white_lane:
        _, _, left_x2, _ = yellow_lane[0][0]
        _, _, right_x2, _ = white_lane[0][0]
        lane_center = (left_x2 + right_x2) / 2
        x_offset = lane_center - mid
    elif yellow_lane:
        x1, _, x2, _ = yellow_lane[0][0]
        x_offset = (x2 - x1) * 0.5  # 保守估计偏差方向
    elif white_lane:
        x1, _, x2, _ = white_lane[0][0]
        x_offset = (x2 - x1) * 0.5
    else:
        return None

    # 将 x_offset 转换为角度误差（假设 y = h/2）
    y_offset = h / 2
    angle_error = math.atan2(x_offset, y_offset)
    steering = pid.control(angle_error)

    # 限制最大转向幅度
    steering = max(-1.0, min(1.0, steering))

    # 转向越大，速度越低
    throttle = max(0.5, 0.63 * (1 - abs(steering)))
    # throttle = max(0.1, 0.6)
    return np.array([steering, throttle])