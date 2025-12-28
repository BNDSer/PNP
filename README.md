相机视频模式
sudo modprobe v4l2loopback video_nr=2
./start_canon_simple.sh
python detect.py