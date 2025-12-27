#!/usr/bin/env python3
"""
相机标定程序
使用9x6棋盘格进行相机内参标定
基于OpenCV的calibrateCamera函数
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

# 棋盘格参数
CHESSBOARD_ROWS = 9  # 内角点行数
CHESSBOARD_COLS = 6  # 内角点列数
SQUARE_SIZE = 20.0   # 方格大小（mm），请根据实际棋盘格修改

# 准备棋盘格的世界坐标（3D点）
def prepare_object_points():
    """准备棋盘格的3D世界坐标点"""
    # 创建 (rows * cols, 3) 的数组，每个点为 (x, y, 0)
    object_points = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)

    # 生成网格点坐标
    for i in range(CHESSBOARD_ROWS):
        for j in range(CHESSBOARD_COLS):
            object_points[i * CHESSBOARD_COLS + j] = [
                j * SQUARE_SIZE,  # x
                i * SQUARE_SIZE,  # y
                0                  # z (平面上)
            ]

    return object_points


def detect_chessboard_corners(frame, show=True):
    """检测棋盘格角点"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray,
        (CHESSBOARD_COLS, CHESSBOARD_ROWS),
        None
    )

    if ret and show:
        # 提高角点精度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 绘制角点
        cv2.drawChessboardCorners(
            frame,
            (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            corners_refined,
            ret
        )

    return ret, corners


def calibrate_camera(object_points, image_points, image_size):
    """执行相机标定"""
    # 标定相机
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    # 计算重投影误差
    mean_error = 0
    for i in range(len(object_points)):
        img_points_reprojected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(image_points[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
        mean_error += error

    mean_error /= len(object_points)

    return ret, camera_matrix, dist_coeffs, mean_error


def save_calibration_results(camera_matrix, dist_coeffs, mean_error, image_size):
    """保存标定结果"""
    # 打印结果
    print("\n" + "=" * 60)
    print("相机标定完成！")
    print("=" * 60)
    print(f"图像尺寸: {image_size}")
    print(f"重投影误差: {mean_error:.4f} 像素")
    print("\n相机内参矩阵:")
    print(camera_matrix)
    print("\n畸变系数:")
    print(dist_coeffs.T)
    print("=" * 60)

    # 保存到文件
    filename = "camera_calibration.txt"
    with open(filename, 'w') as f:
        f.write("相机标定结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"标定时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"棋盘格: {CHESSBOARD_ROWS}x{CHESSBOARD_COLS}, 方格大小: {SQUARE_SIZE}mm\n")
        f.write(f"图像尺寸: {image_size}\n")
        f.write(f"重投影误差: {mean_error:.4f} 像素\n")
        f.write("\n相机内参矩阵 (3x3):\n")
        np.savetxt(f, camera_matrix, fmt='%.6f', delimiter=' ')
        f.write("\n畸变系数 (5x1):\n")
        np.savetxt(f, dist_coeffs.T, fmt='%.6f', delimiter=' ')
        f.write("\n" + "=" * 60 + "\n")
        f.write("\nPython代码格式:\n")
        f.write("import numpy as np\n")
        f.write("\ncamera_matrix = np.array([\n")
        for i in range(3):
            f.write("    [" + ", ".join([f"{camera_matrix[i,j]:.6f}" for j in range(3)]) + "],\n")
        f.write("], dtype=np.float64)\n")
        f.write("\ndist_coeffs = np.array([\n")
        # 确保dist_coeffs是一维的
        dist_flat = dist_coeffs.flatten()
        f.write("    " + ", ".join([f"{dist_flat[i]:.6f}" for i in range(5)]) + "\n")
        f.write("], dtype=np.float64).reshape(-1, 1)\n")

    print(f"\n标定结果已保存到: {filename}")

    # 保存为numpy格式
    np.savez('calibration_data.npz',
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=image_size,
             mean_error=mean_error)
    print("标定数据已保存到: calibration_data.npz")


def main():
    print("=" * 60)
    print("相机标定程序 (9x6棋盘格)")
    print("=" * 60)

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

    print(f"分辨率: {width}x{height}")
    print(f"棋盘格: {CHESSBOARD_ROWS}x{CHESSBOARD_COLS}, 方格大小: {SQUARE_SIZE}mm")

    # 准备3D世界坐标点
    object_points = prepare_object_points()

    # 存储所有采集的图像点和对应的世界坐标点
    object_points_list = []
    image_points_list = []
    saved_images = []

    print("\n操作说明:")
    print("  空格键 : 采集当前图像（至少需要10-20张）")
    print("  U 键   : 撤销最后一次采集")
    print("  C 键   : 执行标定")
    print("  S 键   : 保存当前帧为图片")
    print("  ESC/Q  : 退出")
    print("\n提示:")
    print("  - 从不同角度和距离拍摄棋盘格")
    print("  - 确保棋盘格完全在画面内")
    print("  - 建议采集15-30张图像")
    print("=" * 60)

    frame_count = 0
    captured_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("无法读取帧")
            break

        frame_count += 1

        # 检测棋盘格角点
        found, corners = detect_chessboard_corners(frame.copy())

        # 显示信息
        status_text = f"Frame: {frame_count} | Captured: {captured_count}"
        if found:
            status_text += " [Chessboard Detected!]"
            status_color = (0, 255, 0)
        else:
            status_text += " [No Chessboard]"
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, 'Space:Capture U:Undo C:Calibrate S:Save ESC:Quit',
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示画面
        cv2.imshow('Camera Calibration', frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            print("\n退出标定程序")
            break
        elif key == ord(' '):
            if found:
                # 提高角点精度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # 保存角点和对应的世界坐标
                object_points_list.append(object_points)
                image_points_list.append(corners_refined)
                saved_images.append(frame.copy())

                captured_count += 1
                print(f"\n已采集第 {captured_count} 张图像 (至少需要10张)")

                # 保存原始图像
                img_filename = f'calibration_img_{captured_count}.jpg'
                cv2.imwrite(img_filename, frame)
                print(f"已保存图像: {img_filename}")
            else:
                print("\n未检测到棋盘格，无法采集")
        elif key == ord('u'):
            if captured_count > 0:
                object_points_list.pop()
                image_points_list.pop()
                saved_images.pop()
                captured_count -= 1
                print(f"\n已撤销最后一次采集，剩余 {captured_count} 张")
            else:
                print("\n没有可撤销的图像")
        elif key == ord('c'):
            if captured_count == 0:
                print("\n错误: 还没有采集任何图像！")
                print("请先按空格键采集至少10张不同角度的棋盘格图像")
                continue
            elif captured_count < 10:
                print(f"\n当前采集数量: {captured_count} (建议至少10张)")
                print("建议继续采集更多图像以获得更好的标定效果")
                choice = input("是否继续标定？(y/n): ")
                if choice.lower() != 'y':
                    continue
            else:
                print(f"\n开始标定，使用 {captured_count} 张图像...")

            # 执行标定
            try:
                ret, camera_matrix, dist_coeffs, mean_error = calibrate_camera(
                    object_points_list,
                    image_points_list,
                    (width, height)
                )
            except cv2.error as e:
                print(f"\n标定失败: {e}")
                print("请确保:")
                print("  1. 已采集足够数量的图像 (至少10张)")
                print("  2. 图像中包含完整的棋盘格")
                print("  3. 棋盘格没有被遮挡")
                continue

            if ret:
                # 保存结果
                save_calibration_results(camera_matrix, dist_coeffs, mean_error, (width, height))

                # 畸变校正示例
                print("\n是否查看畸变校正效果？(y/n): ")
                if input().lower() == 'y':
                    # 对最后一幅图像进行畸变校正
                    if saved_images:
                        img = saved_images[-1]
                        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

                        # 并排显示
                        h, w = img.shape[:2]
                        result = np.zeros((h, w * 2, 3), dtype=np.uint8)
                        result[:, :w] = img
                        result[:, w:] = undistorted

                        cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(result, "Undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        cv2.imshow('Undistortion Comparison', result)
                        print("按任意键继续...")
                        cv2.waitKey(0)
                        cv2.destroyWindow('Undistortion Comparison')

                print("\n标定完成！请将生成的 camera_calibration.txt 中的参数更新到 detect.py 中")
                break
            else:
                print("\n标定失败！")
        elif key == ord('s'):
            filename = f'calibration_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"\n已保存: {filename}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print("\n程序结束")


if __name__ == "__main__":
    main()
