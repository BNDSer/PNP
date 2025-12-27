#!/usr/bin/env python3
"""
装甲板识别与6D位姿解算程序
基于传统视觉方法检测装甲板灯条，并使用PNP解算6D位姿
支持GPU加速
"""

import cv2
import numpy as np
import sys
import os

# 尝试导入GPU加速库
try:
    import cupy as cp
    USE_CUPY = True
    print("使用CuPy进行GPU加速")
except ImportError:
    USE_CUPY = False
    print("未安装CuPy，使用CPU计算")

# 检查OpenCV CUDA支持
try:
    # 检查是否有cv2.cuda模块
    if not hasattr(cv2, 'cuda'):
        raise AttributeError("OpenCV编译时未包含CUDA支持")

    # 尝试获取CUDA设备数量
    device_count = cv2.cuda.getCudaEnabledDeviceCount()

    if device_count > 0:
        USE_CUDA = True
        print(f"OpenCV CUDA支持已启用，GPU设备数: {device_count}")

        # 打印GPU设备信息
        try:
            device_info = cv2.cuda.DeviceInfo()
            print(f"  GPU名称: {device_info.name()}")
            print(f"  计算能力: {device_info.majorVersion()}.{device_info.minorVersion()}")
            print(f"  多处理器数: {device_info.numberOfMPs()}")
            print(f"  总内存: {device_info.totalMemory() / 1024 / 1024:.0f} MB")
        except:
            pass
    else:
        USE_CUDA = False
        print("OpenCV CUDA不可用，未检测到GPU设备，使用CPU计算")
except AttributeError:
    USE_CUDA = False
    print("OpenCV CUDA不支持：当前OpenCV版本未编译CUDA模块，使用CPU计算")
except Exception as e:
    USE_CUDA = False
    print(f"OpenCV CUDA不支持 ({e})，使用CPU计算")

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


def get_binary_image(img, threshold=220, use_gpu=False):
    """图像二值化（支持GPU加速）"""
    if use_gpu and USE_CUDA:
        # GPU加速版本
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # GPU上转换灰度
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2GRAY)

        # GPU上二值化
        gpu_binary = cv2.cuda.threshold(gpu_gray, threshold, 255, cv2.THRESH_BINARY)[1]

        # 下载回CPU
        binary = gpu_binary.download()
    else:
        # CPU版本
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

            # 合格的装甲板
            armor_kpts = np.array([
                lights[i].top,
                lights[i].bottom,
                lights[j].bottom,
                lights[j].top
            ], dtype=np.float32)

            armors.append(armor_kpts)

    return armors


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

    # Simplify模式开关（默认关闭）
    # True: 使用简化PNP（取绝对值避免多解）
    # False: 使用标准OpenCV PNP（可能有多个解）
    USE_SIMPLIFY_PNP = True

    # PNP解算超时阈值（毫秒），超过此时间的帧将跳过PNP解算
    PNP_TIMEOUT_MS = 50

    # GPU加速开关
    USE_GPU = True  # 设为True启用GPU加速

    print(f"\n配置:")
    print(f"  Simplify模式: {'开启' if USE_SIMPLIFY_PNP else '关闭'}")
    print(f"  GPU加速: {'开启' if USE_GPU and (USE_CUDA or USE_CUPY) else '关闭/不可用'}")

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
    print("  ESC/Q: 退出")
    print("=" * 60)

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法读取帧")
                break

            frame_count += 1

        # 转为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. 二值化（使用GPU加速）
        binary = get_binary_image(frame_rgb, threshold=220, use_gpu=USE_GPU)

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
                # 获取位置向量
                position = tvec.flatten()

                if USE_SIMPLIFY_PNP:
                    # 简化模式：取绝对值，避免多解问题
                    position = np.abs(position)
                    print(f"装甲板[SIMPLIFY]: 距离={np.linalg.norm(position):.1f}mm, 位置=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
                else:
                    # 标准模式：保持原始值
                    print(f"装甲板[STANDARD]: 距离={np.linalg.norm(position):.1f}mm, 位置=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

                # 绘制3D坐标系
                draw_coordinate_system(frame, armor_kpts, rvec, tvec, camera_matrix, dist_coeffs)

                # 显示距离和位置信息
                distance = np.linalg.norm(position)
                label = f"{distance:.0f}mm"
                cv2.putText(frame, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 显示坐标
                pos_text = f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})"
                cv2.putText(frame, pos_text, tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif pnp_time >= PNP_TIMEOUT_MS:
                # PNP解算超时，跳过位姿解算，只保留矩形框
                print(f"装甲板: PNP解算超时 ({pnp_time:.1f}ms > {PNP_TIMEOUT_MS}ms)，跳过位姿解算")
                # 在矩形框旁显示超时标记
                cv2.putText(frame, 'TIMEOUT', tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # PNP解算失败
                print(f"装甲板: PNP解算失败")

        # 显示检测信息
        mode_text = '[SIMPLIFY]' if USE_SIMPLIFY_PNP else '[STANDARD]'
        gpu_text = '[GPU]' if USE_GPU and (USE_CUDA or USE_CUPY) else '[CPU]'
        info_text = f'Frame: {frame_count} | Lights: {len(lights)} | Armors: {len(armors)} {mode_text} {gpu_text}'
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

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print("\n程序结束")
    print(f"总共处理了 {frame_count} 帧")


if __name__ == "__main__":
    main()
