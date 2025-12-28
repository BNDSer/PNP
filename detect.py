#!/usr/bin/env python3
"""
装甲板识别与6D位姿解算程序
基于传统视觉方法检测装甲板灯条，并使用PNP解算6D位姿
"""

import cv2
import numpy as np
import sys
import os
from collections import deque
from copy import deepcopy

# 装甲板实际尺寸（单位：mm）
'''ARMOR_SMALL = np.array([
    [-67.5, -27.5, 0],  # 左上
    [-67.5, 27.5, 0],   # 左下
    [67.5, 27.5, 0],    # 右下
    [67.5, -27.5, 0]    # 右上
], dtype=np.float64)'''

ARMOR_SMALL = np.array([
    [-22.38, -12.34, 0],  # 左上
    [-22.38, 12.34, 0],   # 左下
    [22.38, 12.34, 0],    # 右下
    [22.38, -12.34, 0]    # 右上
], dtype=np.float64)

ARMOR_BIG = np.array([
    [-112.5, -27.5, 0],  # 左上
    [-112.5, 27.5, 0],   # 左下
    [112.5, 27.5, 0],    # 右下
    [112.5, -27.5, 0]    # 右上
], dtype=np.float64)

# Simplify模式开关（默认开启）
# True: 简化模式，始终认为装甲板面朝相机，灯条正立
# False: 标准模式，保持原始检测值
USE_SIMPLIFY_PNP = True

# 历史帧最大保存数量
MAX_HISTORY_FRAMES = 10


class KalmanFilter:
    """卡尔曼滤波器类，用于平滑装甲板的位姿估计"""
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        # 状态向量: [x, y, z, vx, vy, vz] (位置和速度)
        self.state_dim = 6
        self.measurement_dim = 3

        # 初始化状态
        self.state = np.zeros((self.state_dim, 1))
        self.initialized = False

        # 状态转移矩阵 (匀速模型)
        self.F = np.eye(self.state_dim, dtype=np.float64)
        dt = 1.0  # 假设时间步长为1
        for i in range(3):
            self.F[i, i+3] = dt

        # 测量矩阵 (只观测位置)
        self.H = np.zeros((self.measurement_dim, self.state_dim), dtype=np.float64)
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z

        # 过程噪声协方差
        self.Q = process_noise * np.eye(self.state_dim, dtype=np.float64)

        # 测量噪声协方差
        self.R = measurement_noise * np.eye(self.measurement_dim, dtype=np.float64)

        # 状态协方差
        self.P = np.eye(self.state_dim, dtype=np.float64)

    def predict(self):
        """预测步骤"""
        if not self.initialized:
            return None

        # 状态预测: x = F * x
        self.state = self.F @ self.state

        # 协方差预测: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:3].flatten()  # 返回预测位置

    def update(self, measurement):
        """更新步骤

        Args:
            measurement: 3D位置测量值 [x, y, z]
        """
        measurement = np.array(measurement).reshape(-1, 1)

        if not self.initialized:
            # 首次初始化
            self.state[:3] = measurement
            self.state[3:] = 0  # 初始速度设为0
            self.initialized = True
            return measurement.flatten()

        # 计算卡尔曼增益: K = P * H^T * (H * P * H^T + R)^-1
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态: x = x + K * (z - H * x)
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation

        # 更新协方差: P = (I - K * H) * P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:3].flatten()

    def get_position(self):
        """获取当前估计位置"""
        if not self.initialized:
            return None
        return self.state[:3].flatten()


class ArmorTracker:
    """装甲板跟踪器，管理历史帧和卡尔曼滤波"""
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.reset()

    def reset(self):
        """重置跟踪器"""
        self.kalman_filter = KalmanFilter()
        self.history_frames = deque(maxlen=self.max_history)
        self.no_detection_count = 0
        self.last_valid_detection = None
        self.last_valid_frame_image = None
        self.last_valid_frame_data = None

    def add_detection(self, position, rvec, tvec, armor_kpts, frame_image):
        """添加新的检测结果

        Args:
            position: 3D位置
            rvec: 旋转向量
            tvec: 平移向量
            armor_kpts: 装甲板关键点
            frame_image: 帧图像（带绘制的检测结果）
        """
        # 使用卡尔曼滤波更新位置估计
        filtered_position = self.kalman_filter.update(position)

        # 保存检测结果和历史图像
        self.last_valid_detection = {
            'position': filtered_position,
            'rvec': rvec.copy() if rvec is not None else None,
            'tvec': tvec.copy() if tvec is not None else None,
            'armor_kpts': armor_kpts.copy()
        }

        self.last_valid_frame_image = frame_image.copy()

        # 保存历史帧数据
        self.last_valid_frame_data = {
            'position': filtered_position.copy(),
            'rvec': rvec.copy() if rvec is not None else None,
            'tvec': tvec.copy() if tvec is not None else None,
            'armor_kpts': armor_kpts.copy(),
            'frame_count': len(self.history_frames)
        }

        self.history_frames.append(self.last_valid_frame_data)

        # 重置无检测计数器
        self.no_detection_count = 0

        return filtered_position

    def predict(self):
        """预测当前帧的位置

        Returns:
            (has_valid_data, data_dict)
            has_valid_data: 是否有有效数据
            data_dict: 包含position, rvec, tvec, armor_kpts, frame_image
        """
        # 卡尔曼滤波预测
        predicted_position = self.kalman_filter.predict()

        if predicted_position is None:
            return False, None

        # 检查是否超过最大无检测帧数
        if self.no_detection_count >= self.max_history:
            return False, None

        # 增加无检测计数
        self.no_detection_count += 1

        # 使用上一次的有效检测结果
        if self.last_valid_detection is None:
            return False, None

        # 返回带有预测位置的检测结果
        result = {
            'position': predicted_position,
            'rvec': self.last_valid_detection['rvec'],
            'tvec': self.last_valid_detection['tvec'],
            'armor_kpts': self.last_valid_detection['armor_kpts'],
            'frame_image': self.last_valid_frame_image.copy() if self.last_valid_frame_image is not None else None,
            'is_predicted': True,
            'no_detection_count': self.no_detection_count
        }

        return True, result

    def get_filtered_position(self):
        """获取当前滤波后的位置"""
        return self.kalman_filter.get_position()

    def has_valid_data(self):
        """是否有有效的历史数据"""
        return self.last_valid_detection is not None and self.no_detection_count < self.max_history


def load_camera_calibration(calibration_file='calibration_data.npz'):
    """
    从标定文件加载相机内参
    Args:
        calibration_file: 标定数据文件路径
    Returns:
        camera_matrix, dist_coeffs 或默认值（如果文件不存在）
    """
    if os.path.exists(calibration_file):
        try:
            data = np.load(calibration_file)
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            mean_error = float(data['mean_error'])

            print("=" * 60)
            print("已加载相机标定数据")
            print(f"标定文件: {calibration_file}")
            print(f"重投影误差: {mean_error:.4f} 像素")
            print("相机内参矩阵:")
            print(camera_matrix)
            print("畸变系数:")
            print(dist_coeffs.T)
            print("=" * 60)

            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"警告: 无法读取标定文件 ({e})")
            print("使用默认估计值")
    else:
        print(f"警告: 标定文件不存在 ({calibration_file})")
        print("使用默认估计值")
        print("请先运行: python calibrate.py 进行标定")

    # 返回默认估计值
    return None, None


class Light:
    """灯条类"""
    def __init__(self, contour):
        if len(contour) < 5:
            # 点数太少，用最小外接矩形
            rect = cv2.minAreaRect(contour)
            self.top, self.bottom = self._get_points_by_rect(rect)
        else:
            # 椭圆拟合
            ellipse = cv2.fitEllipse(contour)
            rect = cv2.minAreaRect(contour)
            self.top, self.bottom = self._get_points_by_ellipse(ellipse, rect)

        # SIMPLIFY模式：确保灯条倒立（top的y坐标始终大于bottom的y坐标，即top在bottom之下）
        if USE_SIMPLIFY_PNP and self.top[1] < self.bottom[1]:
            self.top, self.bottom = self.bottom, self.top

        self.center = (self.top + self.bottom) / 2.0
        self.length = np.linalg.norm(self.bottom - self.top)

        # 计算宽度
        box = cv2.boxPoints(rect)
        box_sorted = sorted(box, key=lambda p: p[1])
        self.width = (np.linalg.norm(box_sorted[0] - box_sorted[1]) +
                     np.linalg.norm(box_sorted[2] - box_sorted[3])) / 2.0

        self.contour = contour
        self.bounding_rect = cv2.boundingRect(contour)

    def _get_points_by_rect(self, rect):
        """通过最小外接矩形获取上下端点"""
        center, size, angle = rect
        width, height = size

        if angle > 45:
            angle -= 90
            width, height = height, width
        elif angle < -45:
            angle += 90
            width, height = height, width

        rad = np.deg2rad(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        # 上下端点
        top = center + np.array([sin_a * height / 2, -cos_a * height / 2])
        bottom = center - np.array([sin_a * height / 2, -cos_a * height / 2])

        return top, bottom

    def _get_points_by_ellipse(self, ellipse, rect):
        """通过椭圆拟合获取上下端点"""
        center, axes, angle = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)

        # 如果长轴和短轴差距不大，或短轴太小，用矩形方法
        if minor_axis < 1.0 or major_axis / minor_axis < 2.0:
            return self._get_points_by_rect(rect)

        # 使用椭圆中心轴方向
        rad = np.deg2rad(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        top = center + np.array([sin_a * major_axis / 2, -cos_a * major_axis / 2])
        bottom = center - np.array([sin_a * major_axis / 2, -cos_a * major_axis / 2])

        return top, bottom


def get_binary_image(img, threshold=220):
    """图像二值化"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def detect_light_color(img, light_contour, rect):
    """检测灯条颜色（红/蓝）"""
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]

    count_red = 0
    count_blue = 0

    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            # 判断点是否在轮廓内
            point = (j + x, i + y)
            if cv2.pointPolygonTest(light_contour, point, False) >= 0:
                count_red += roi[i, j, 2]
                count_blue += roi[i, j, 0]

    return 0 if count_red > count_blue else 1  # 0=红色, 1=蓝色


def get_valid_lights(binary, src):
    """筛选有效灯条"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lights = []

    for cnt in contours:
        # 点数太少，跳过
        if len(cnt) < 5:
            continue

        try:
            light = Light(cnt)
        except Exception as e:
            # 创建灯条对象时出错（如NaN值），跳过该轮廓
            continue

        # 检查NaN值
        if np.any(np.isnan(light.top)) or np.any(np.isnan(light.bottom)) or \
           np.any(np.isnan(light.center)) or np.isnan(light.length) or np.isnan(light.width):
            continue

        # 灯条在边缘，跳过
        x, y, w, h = light.bounding_rect
        if x < 0 or x + w >= src.shape[1] or y < 0 or y + h >= src.shape[0]:
            continue

        # 宽高比检查
        ratio = light.width / light.length
        if ratio < 0.05 or ratio > 0.6:
            continue

        # 倾斜角度检查
        direction = light.top - light.bottom
        tilt_angle = np.abs(np.arctan2(np.abs(direction[0]), np.abs(direction[1])))
        if tilt_angle > np.deg2rad(45):
            continue

        # 颜色检测
        color = detect_light_color(src, cnt, light.bounding_rect)
        light.color = color

        lights.append(light)

    # 从左到右排序
    lights.sort(key=lambda l: l.center[0])
    return lights


def contain_light(lights, i, j):
    """判断两个灯条中间是否夹着其他灯条"""
    l1 = lights[i]
    l2 = lights[j]
    contour = np.array([l1.top, l1.bottom, l2.bottom, l2.top], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)

    for idx in range(i + 1, j):
        l = lights[idx]
        # 检查灯条的关键点是否在矩形内
        for point in [l.top, l.center, l.bottom]:
            px, py = int(point[0]), int(point[1])
            if x <= px <= x + w and y <= py <= y + h:
                return True
    return False


def get_matched_armors(lights):
    """匹配装甲板"""
    armors = []

    for i in range(len(lights)):
        for j in range(i + 1, len(lights)):
            # 灯条间不能有其他灯条
            if contain_light(lights, i, j):
                continue

            # 长度相似性检查
            shorter = min(lights[i].length, lights[j].length)
            longer = max(lights[i].length, lights[j].length)
            if shorter / longer < 0.5:
                continue

            # 宽高比检查
            w = np.linalg.norm(lights[i].center - lights[j].center)
            h = (lights[i].length + lights[j].length) / 2
            if w / h < 1.0 or w / h > 5.0:
                continue

            # 倾斜角度检查
            direction = lights[j].center - lights[i].center
            angle = np.abs(np.arctan2(direction[1], direction[0]))
            if np.abs(angle) > np.deg2rad(60):
                continue

            # 生成装甲板角点
            armor_kpts = np.array([
                lights[i].top,
                lights[i].bottom,
                lights[j].bottom,
                lights[j].top
            ], dtype=np.float32)

            # 检测矩形框是否交叉，如果交叉则修正
            armor_kpts = fix_crossed_armor(armor_kpts, lights[i], lights[j])

            armors.append(armor_kpts)

    return armors


def fix_crossed_armor(kpts, light1, light2):
    """检测并修正交叉的装甲板角点

    检测四边形是否自相交，如果相交则交换一个灯条的top和bottom
    """
    # 检测线段是否交叉：(0,1)与(2,3)或(1,2)与(3,0)
    if lines_intersect(kpts[0], kpts[1], kpts[2], kpts[3]) or \
       lines_intersect(kpts[1], kpts[2], kpts[3], kpts[0]):

        # 尝试交换第二个灯条的top和bottom
        kpts_fixed = np.array([
            light1.top,
            light1.bottom,
            light2.top,
            light2.bottom
        ], dtype=np.float32)

        # 检查修正后是否仍然交叉
        if not (lines_intersect(kpts_fixed[0], kpts_fixed[1], kpts_fixed[2], kpts_fixed[3]) or \
               lines_intersect(kpts_fixed[1], kpts_fixed[2], kpts_fixed[3], kpts_fixed[0])):
            return kpts_fixed

    return kpts


def lines_intersect(p1, p2, p3, p4):
    """检测两条线段是否相交

    使用向量叉积方法检测线段(p1,p2)和(p3,p4)是否相交
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def is_big_armor(kpts):
    """判断是否为大装甲板"""
    # 计算宽度
    width_left = np.linalg.norm(kpts[0] - kpts[1])
    width_right = np.linalg.norm(kpts[2] - kpts[3])
    width = (width_left + width_right) / 2

    # 计算高度
    height_left = np.linalg.norm(kpts[0] - kpts[3])
    height_right = np.linalg.norm(kpts[1] - kpts[2])
    height = (height_left + height_right) / 2

    ratio = width / height
    # 大装甲板宽高比约4:1，小装甲板约2.5:1
    return ratio > 3.0


def draw_coordinate_system(img, corners, rvec, tvec, camera_matrix, dist_coeffs):
    """绘制3D坐标系"""
    # 定义3D坐标轴（单位：mm）
    axis_length = 100  # 100mm
    axis_3d = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],  # X轴 - 红色
        [0, axis_length, 0],  # Y轴 - 绿色
        [0, 0, axis_length]   # Z轴 - 蓝色
    ], dtype=np.float64)

    # 投影到2D图像平面
    axis_2d, _ = cv2.projectPoints(
        axis_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    axis_2d = axis_2d.reshape(-1, 2).astype(np.int32)

    # 绘制坐标轴
    origin = tuple(axis_2d[0])
    cv2.line(img, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X轴 - 红色
    cv2.line(img, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y轴 - 绿色
    cv2.line(img, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z轴 - 蓝色

    # 绘制标签
    cv2.putText(img, 'X', tuple(axis_2d[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, 'Y', tuple(axis_2d[2] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, 'Z', tuple(axis_2d[3] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def main():
    print("=" * 60)
    print("装甲板识别与6D位姿解算程序")
    print("=" * 60)

    # PNP解算超时阈值（毫秒），超过此时间的帧将跳过PNP解算
    PNP_TIMEOUT_MS = 50

    # 打开摄像头
    camera_index = 2
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])

    print(f"\n正在打开相机设备 /dev/video{camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"\n错误: 无法打开/dev/video{camera_index}")
        return

    # 获取相机参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n分辨率: {width}x{height}")

    # 尝试从标定文件加载相机内参
    camera_matrix, dist_coeffs = load_camera_calibration('calibration_data.npz')

    # 如果标定文件不存在，使用估计值
    if camera_matrix is None:
        print("\n使用相机内参估计值...")
        fx, fy = width * 0.8, height * 0.8  # 焦距估计
        cx, cy = width / 2, height / 2     # 主点
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    print("\n操作说明:")
    print("  空格键: 暂停/继续")
    print("  S 键: 保存当前帧")
    print("  R 键: 重置跟踪器")
    print("  ESC/Q: 退出")
    print("=" * 60)

    paused = False
    frame_count = 0

    # 创建装甲板跟踪器
    armor_tracker = ArmorTracker(max_history=MAX_HISTORY_FRAMES)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法读取帧")
                break

            frame_count += 1

        # 转为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. 二值化
        binary = get_binary_image(frame_rgb, threshold=220)

        # 2. 检测灯条
        lights = get_valid_lights(binary, frame_rgb)

        # 3. 绘制灯条
        for light in lights:
            color = (255, 0, 0) if light.color == 0 else (0, 0, 255)  # 红/蓝
            top = tuple(map(int, light.top))
            bottom = tuple(map(int, light.bottom))
            center = tuple(map(int, light.center))
            cv2.line(frame, top, bottom, color, 2)
            cv2.circle(frame, center, 3, color, -1)

        # 4. 匹配装甲板
        armors = get_matched_armors(lights)

        # 5. 对每个装甲板进行PNP解算
        detected_armor = False  # 标记当前帧是否检测到装甲板

        for armor_kpts in armors:
            # 绘制装甲板轮廓
            pts = armor_kpts.reshape(-1, 2).astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # 判断大小
            is_big = is_big_armor(armor_kpts)
            object_points = ARMOR_BIG if is_big else ARMOR_SMALL

            # 记录PNP解算开始时间
            pnp_start_time = cv2.getTickCount()

            # 标准OpenCV PNP解算
            success, rvec, tvec = cv2.solvePnP(
                object_points, armor_kpts,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE
            )

            # 计算PNP解算耗时（毫秒）
            pnp_time = (cv2.getTickCount() - pnp_start_time) * 1000 / cv2.getTickFrequency()

            if success and pnp_time < PNP_TIMEOUT_MS:
                # PNP解算成功且未超时
                detected_armor = True

                # 获取位置向量
                position = tvec.flatten()

                if USE_SIMPLIFY_PNP:
                    # 简化模式：取绝对值，避免多解问题
                    position = np.abs(position)

                # 使用卡尔曼滤波平滑位置估计
                filtered_position = armor_tracker.add_detection(position, rvec, tvec, armor_kpts, frame)

                # 打印信息（显示滤波后的结果）
                mode_str = '[SIMPLIFY]' if USE_SIMPLIFY_PNP else '[STANDARD]'
                print(f"装甲板{mode_str}[检测]: 距离={np.linalg.norm(filtered_position):.1f}mm, "
                      f"原始=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), "
                      f"滤波=({filtered_position[0]:.1f}, {filtered_position[1]:.1f}, {filtered_position[2]:.1f})")

                # 使用滤波后的位置进行显示
                display_position = filtered_position

                # 绘制3D坐标系
                draw_coordinate_system(frame, armor_kpts, rvec, tvec, camera_matrix, dist_coeffs)

                # 显示距离和位置信息
                distance = np.linalg.norm(display_position)
                label = f"{distance:.0f}mm"
                cv2.putText(frame, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 显示坐标
                pos_text = f"({display_position[0]:.0f}, {display_position[1]:.0f}, {display_position[2]:.0f})"
                cv2.putText(frame, pos_text, tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 显示卡尔曼滤波标记
                cv2.putText(frame, 'KF', tuple(pts[0] + np.array([0, 20])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            elif pnp_time >= PNP_TIMEOUT_MS:
                # PNP解算超时，跳过位姿解算，只保留矩形框
                print(f"装甲板: PNP解算超时 ({pnp_time:.1f}ms > {PNP_TIMEOUT_MS}ms)，跳过位姿解算")
                # 在矩形框旁显示超时标记
                cv2.putText(frame, 'TIMEOUT', tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # PNP解算失败
                print(f"装甲板: PNP解算失败")

        # 6. 如果当前帧没有检测到装甲板，尝试使用历史帧数据和卡尔曼预测
        if not detected_armor and armor_tracker.has_valid_data():
            has_prediction, prediction_data = armor_tracker.predict()

            if has_prediction and prediction_data is not None:
                # 使用预测的历史数据
                position = prediction_data['position']
                no_detect_count = prediction_data['no_detection_count']
                armor_kpts = prediction_data['armor_kpts']
                rvec = prediction_data['rvec']
                tvec = prediction_data['tvec']

                print(f"装甲板[预测-{no_detect_count}/{MAX_HISTORY_FRAMES}]: "
                      f"距离={np.linalg.norm(position):.1f}mm, "
                      f"位置=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

                # 绘制装甲板轮廓（使用历史关键点）
                if armor_kpts is not None:
                    pts = armor_kpts.reshape(-1, 2).astype(int)
                    # 使用虚线效果绘制预测的装甲板
                    cv2.polylines(frame, [pts], True, (0, 165, 255), 2)

                    # 绘制3D坐标系
                    if rvec is not None and tvec is not None:
                        draw_coordinate_system(frame, armor_kpts, rvec, tvec, camera_matrix, dist_coeffs)

                        # 显示距离和位置信息
                        distance = np.linalg.norm(position)
                        label = f"P:{distance:.0f}mm"
                        cv2.putText(frame, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                        # 显示坐标
                        pos_text = f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})"
                        cv2.putText(frame, pos_text, tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                # 在当前帧上显示预测信息
                predict_text = f"PREDICTED ({no_detect_count}/{MAX_HISTORY_FRAMES})"
                cv2.putText(frame, predict_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 165, 255), 2)

        # 如果连续多帧没有检测，显示警告
        if not detected_armor and armor_tracker.no_detection_count >= MAX_HISTORY_FRAMES:
            cv2.putText(frame, 'NO DETECTION - TRACKING LOST',
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)

        # 显示检测信息
        mode_text = '[SIMPLIFY]' if USE_SIMPLIFY_PNP else '[STANDARD]'
        info_text = f'Frame: {frame_count} | Lights: {len(lights)} | Armors: {len(armors)} {mode_text}'
        if paused:
            info_text += ' [PAUSED]'

        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Space:Pause S:Save ESC:Quit', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示画面
        cv2.imshow('Armor Detection', frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'暂停' if paused else '继续'}")
        elif key == ord('s'):
            filename = f'armor_detection_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
        elif key == ord('r'):
            armor_tracker.reset()
            print("跟踪器已重置")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print("\n程序结束")
    print(f"总共处理了 {frame_count} 帧")


if __name__ == "__main__":
    main()
