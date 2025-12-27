import cv2
import subprocess

def get_supported_resolution(cap):
    """自动检测摄像头支持的最高分辨率"""
    # 常见的高分辨率
    resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 2K
        (1920, 1080),  # 1080p
        (1600, 1200),  # 1200p
        (1280, 960),   # 960p
        (1280, 720),   # 720p
    ]

    print("正在检测支持的最高分辨率...")

    # 尝试不同的后端和像素格式组合
    # 首先尝试MJPEG格式（压缩格式，支持更高分辨率）
    print("尝试方法1: MJPEG压缩格式")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    best_width, best_height = 0, 0
    best_format = "Unknown"

    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 读取FOURCC
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        print(f"尝试 {width}x{height} -> 实际: {actual_width}x{actual_height} ({fourcc_str})")

        if actual_width >= best_width and actual_height >= best_height:
            best_width = actual_width
            best_height = actual_height
            best_format = fourcc_str

    print(f"\n最佳分辨率: {best_width}x{best_height} (格式: {best_format})")

    # 如果还是没得到1080p，尝试YUYV格式
    if best_width < 1920:
        print("\n尝试方法2: YUYV原始格式")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))

        for width, height in [(1920, 1080), (1280, 960)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"尝试 {width}x{height} -> 实际: {actual_width}x{actual_height}")

            if actual_width > best_width:
                best_width = actual_width
                best_height = actual_height

    # 设置为最佳分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)

    return best_width, best_height

def setup_exposure(cap):
    """设置曝光，尝试多种方法"""
    print("\n正在检查曝光控制...")

    print("尝试OpenCV标准UVC模式")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    result1 = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    print(f"  CAP_PROP_AUTO_EXPOSURE = {result1}")

    cap.set(cv2.CAP_PROP_EXPOSURE, -7)
    result2 = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"  CAP_PROP_EXPOSURE = {result2}")

    print("\n尝试v4l2-ctl工具检查可用控制")
    try:
        result = subprocess.run(['which', 'v4l2-ctl'], capture_output=True, text=True)
        if result.returncode == 0:
            result = subprocess.run(['v4l2-ctl', '-d', '/dev/video0', '--list-ctrls'],
                                  capture_output=True, text=True)
            if 'gain' in result.stdout.lower():
                print("  ✓ 支持gain控制（将用于调节明暗）")
            if 'exposure' in result.stdout.lower():
                print("  ✓ 支持exposure控制")
            else:
                print("  ✗ 不支持exposure控制（海康威视常见问题）")
        else:
            print("  v4l2-ctl未安装")
    except Exception as e:
        print(f"  检查失败: {e}")

    return 0, 255  # gain范围

def set_gain_v4l2(device, value):
    """使用v4l2-ctl设置增益值（替代曝光控制）"""
    try:
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'gain={value}'],
                      capture_output=True, timeout=1)
        return True
    except:
        return False

def set_brightness_v4l2(device, value):
    """使用v4l2-ctl设置亮度"""
    try:
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'brightness={value}'],
                      capture_output=True, timeout=1)
        return True
    except:
        return False

def main():
    print("=" * 60)
    print("海康威视USB摄像头测试程序")
    print("=" * 60)

    # 使用直接模式（不使用MJPEG）
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("无法打开摄像头")
        print("请检查:")
        print("  - 摄像头是否连接")
        print("  - 是否有其他程序占用摄像头")
        print("  - 尝试使用 ls /dev/video* 查看设备")
        return

    # 自动检测最高分辨率
    width, height = get_supported_resolution(cap)

    # 尝试设置更高的帧率和缓冲区
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，降低延迟

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    backend = cap.getBackendName()

    print("\n" + "=" * 60)
    print("摄像头信息:")
    print(f"  分辨率: {actual_width}x{actual_height}")
    print(f"  帧率: {actual_fps}")
    print(f"  后端: {backend}")
    print("=" * 60)

    # 设置曝光（海康威视用gain替代）
    min_exposure, max_exposure = 0, 255  # gain范围
    default_exposure = 128  # gain默认值
    exposure = default_exposure

    print("\n注意: 海康威视摄像头不支持标准exposure控制")
    print("将使用gain（增益）来调节画面明暗")
    print(f"初始gain值: {exposure}")

    # 创建窗口
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    print("\n操作说明:")
    print("  ↑/↓ : 微调增益 (+10/-10)")
    print("  ←/→ : 大幅调节 (+50/-50)")
    print("  R   : 重置增益")
    print("  I   : 显示摄像头信息")
    print("  ESC/Q: 退出")
    print("=" * 60)
    print("当前使用: v4l2-ctl控制gain（增益）")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("无法读取帧")
            break

        # 显示信息
        info_text = f'{actual_width}x{actual_height} | Gain: {exposure}'

        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Arrows: Adjust Gain | R=Reset | I=Info', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示画面
        cv2.imshow('Camera', frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            exposure = default_exposure
            set_gain_v4l2('/dev/video0', exposure)
            print(f"\n重置增益: {exposure}")
        elif key == ord('i'):
            print("\n" + "=" * 60)
            print("摄像头参数:")
            print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
            print(f"  亮度: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
            print(f"  对比度: {cap.get(cv2.CAP_PROP_CONTRAST)}")
            print(f"  饱和度: {cap.get(cv2.CAP_PROP_SATURATION)}")
            print("  注意: 海康威视摄像头不支持标准曝光控制")
            print("  使用gain调节画面明暗（0=暗，255=亮）")
            print("=" * 60)
        else:
            if key == 81:  # ←
                exposure = max(min_exposure, exposure - 50)
                set_gain_v4l2('/dev/video0', exposure)
                print(f"增益: {exposure}")
            elif key == 82:  # ↑
                exposure = min(max_exposure, exposure + 10)
                set_gain_v4l2('/dev/video0', exposure)
                print(f"增益: {exposure}")
            elif key == 83:  # →
                exposure = min(max_exposure, exposure + 50)
                set_gain_v4l2('/dev/video0', exposure)
                print(f"增益: {exposure}")
            elif key == 84:  # ↓
                exposure = max(min_exposure, exposure - 10)
                set_gain_v4l2('/dev/video0', exposure)
                print(f"增益: {exposure}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print("\n程序结束")
    print("\n如果需要完整的相机控制（曝光时间、增益等），建议:")
    print("1. 使用海康威视官方SDK: https://www.hikvision.com/cn/tools_779.html")
    print("2. 或使用Aravis库（支持GigE工业相机）")

if __name__ == "__main__":
    main()
