#!/bin/bash
# 简化版佳能相机启动脚本
# 请先确保v4l2loopback已加载

echo "正在检查v4l2loopback模块..."
if ! lsmod | grep -q v4l2loopback; then
    echo "错误: v4l2loopback模块未加载"
    echo "请先运行: sudo modprobe v4l2loopback video_nr=2"
    exit 1
fi

echo "正在检查/dev/video2..."
if [ ! -e /dev/video2 ]; then
    echo "错误: /dev/video2 不存在"
    echo "请先运行: sudo modprobe v4l2loopback video_nr=2"
    exit 1
fi

echo "正在停止所有gphoto2进程..."
killall -9 gphoto2 gvfs-gphoto2-volume-monitor gvfsd-gphoto2 2>/dev/null
systemctl --user stop gvfs-gphoto2-volume-monitor.service 2>/dev/null
sleep 2

echo "正在启动佳能相机视频转发..."
echo "相机将输出到 /dev/video2"
echo ""
echo "提示："
echo "  - 确保相机设置为视频模式"
echo "  - 可以在相机上调整曝光参数"
echo "  - 按 Ctrl+C 停止"
echo ""

# 启动视频转发
gphoto2 --stdout --capture-movie | \
ffmpeg -y -i - \
    -vcodec rawvideo \
    -pix_fmt yuv420p \
    -f v4l2 \
    /dev/video2
