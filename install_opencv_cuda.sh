#!/bin/bash
# 安装支持CUDA的OpenCV脚本

echo "========================================="
echo "安装支持CUDA的OpenCV"
echo "========================================="

# 激活conda环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pnp

# 卸载现有opencv
echo "1. 卸载现有OpenCV..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

# 根据CUDA版本安装对应的OpenCV
echo "2. 检测CUDA版本..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
echo "   检测到CUDA $CUDA_VERSION"

echo "3. 安装支持CUDA的OpenCV..."
if [ "$CUDA_VERSION" = "12" ]; then
    pip install opencv-contrib-python-cuda12x
elif [ "$CUDA_VERSION" = "11" ]; then
    pip install opencv-contrib-python-cuda11x
else
    echo "   不支持的CUDA版本，尝试安装CUDA 12版本..."
    pip install opencv-contrib-python-cuda12x
fi

echo ""
echo "4. 验证安装..."
python -c "
import cv2
print(f'OpenCV版本: {cv2.__version__}')
print(f'GPU设备数: {cv2.cuda.getCudaEnabledDeviceCount()}')
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print('✅ OpenCV CUDA支持已启用！')
else:
    print('❌ 仍然没有检测到GPU设备')
"

echo ""
echo "========================================="
echo "安装完成！"
echo "========================================="
