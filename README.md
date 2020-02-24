# Helmet-Detection-Track

## 安全帽检测+追踪

## 环境
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

检测部分yolov3,追踪部分KCF算法

## 使用

使用训练好的权重trained_weights_stage_1.h5,置于models/trained_weights_stage_1.h5

（百度云https://pan.baidu.com/s/1naqCQHYTbT2RMOgxxBwong 密码:q1a4）


```
python traffic_main.py
```
其中：

config,py 中更改相关设置

保存的图片存放于vis文件夹中，路径可在config.py中更改

make_video.py将vis中图片合成视频


