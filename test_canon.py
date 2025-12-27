#!/usr/bin/env python3
"""
佳能相机USB摄像头调用程序
使用gphoto2和v4l2loopback将佳能相机作为USB摄像头设备

安装步骤（见代码后的说明）：
1. 安装gphoto2和v4l2loopback
2. 加载v4l2loopback模块
3. 启动gphoto2视频转发
4. 运行此脚本读取/dev/video0
"""

import cv2
import sys

def main():
    print("=" * 60)
    print("佳能相机USB摄像头程序")
    print("=" * 60)

    # 尝试打开摄像头设备
    # v4l2loopback默认创建/dev/video0
    camera_index = 2

    # 如果有多个摄像头，可以尝试不同的索引
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])

    print(f"\n正在打开相机设备 /dev/video{camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"\n错误: 无法打开/dev/video{camera_index}")
        print("\n请检查:")
        print("1. 相机是否通过USB连接到电脑")
        print("2. 是否已安装并配置gphoto2和v4l2loopback")
        print("3. 是否已启动gphoto2视频转发（见下方命令）")
        print("4. 尝试运行: ls /dev/video* 查看可用设备")
        print("\n可以尝试指定其他摄像头索引:")
        print(f"  python {sys.argv[0]} 1")
        print(f"  python {sys.argv[0]} 2")
        return

    # 获取相机参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    backend = cap.getBackendName()

    print("\n" + "=" * 60)
    print("相机信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps}")
    print(f"  后端: {backend}")
    print("=" * 60)

    # 创建窗口
    window_name = 'Canon Camera'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n程序运行中...")
    print("操作说明:")
    print("  空格键 : 暂停/继续")
    print("  S 键   : 保存当前帧")
    print("  ESC/Q  : 退出")
    print("\n提示: 请直接在相机上调整曝光、ISO等参数")

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("\n错误: 无法读取帧，相机可能已断开")
                break

            frame_count += 1

        # 显示信息
        info_text = f'{width}x{height} | Frame: {frame_count}'
        if paused:
            info_text += ' [PAUSED]'

        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Space:Pause S:Save ESC:Quit', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示画面
        cv2.imshow(window_name, frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'暂停' if paused else '继续'}")
        elif key == ord('s'):
            filename = f'canon_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print("\n程序结束")
    print(f"总共处理了 {frame_count} 帧")

if __name__ == "__main__":
    main()
