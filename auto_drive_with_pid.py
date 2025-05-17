import random
import time

import cv2
import numpy as np
import math
import gym
import gym_donkeycar
import random

from tools import ROI, get_line, get_action, detect_line

from PID import PIDController





def main():
    # 设置模拟器环境
    env = gym.make("donkey-generated-roads-v0")
    # 重置当前场景
    obv = env.reset()

    # 初始化PID控制器
    pid = PIDController()

    # 开始启动
    action = np.array([0, 0.5])
    obv, reward, done, info = env.step(action)
    # 获取图像
    frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    try:
        while True:
            # ① 颜色空间转换
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # ②设置黄色颜色阈值
            lower_yellow = np.array([15, 40, 40])
            upper_yellow = np.array([45, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            # 设置白色颜色阈值
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            # ③Canny边缘轮廓提取
            yellow_edge = cv2.Canny(yellow_mask, 200, 400)
            white_edge = cv2.Canny(white_mask, 200, 400)

            # ④ROI感兴趣区域提取
            yellow_roi = ROI(yellow_edge, 3)
            # cv2.imwrite('yellow_mask.jpg', yellow_roi)
            white_roi = ROI(white_edge, 4)
            # cv2.imwrite('white_mask.jpg', white_roi)

            # ⑤基于霍夫变换的线段检测
            yellow_lane = detect_line(yellow_roi, 'left')
            white_lane = detect_line(white_roi, 'right')

            # ⑥PID控制器获取转向值和油门
            action = get_action(h, w, yellow_lane, white_lane, pid)
            if action is None:
                time.sleep(1)
                break
            # else:
            # 执行动作
            print(action)
            # ⑦执行动作
            obv, reward, done, info = env.step(action)
            # 重新获取图像
            frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)
    except KeyboardInterrupt:
        # CTRL+C暂停
        pass
    obv = env.reset()

if __name__ == '__main__':
    main()
