# 参考

[视频抠图](https://github.com/FeiGeChuanShu/ncnn_Android_RobustVideoMatting)

[FaceDetect-FaceLandmark](https://github.com/hzq-zjm/FaceDetect-FaceLandmark)

[pfld-ncnn](https://github.com/Hsintao/pfld-ncnn)

[pfld-ncnn](https://github.com/nilseuropa/pfld_ncnn)

[PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)

[Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)

[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

[QT+ncnn实现人脸检测及关键点](https://zhuanlan.zhihu.com/p/370608155)

[人脸检测之Ultra-Light-Fast-Generic-Face-Detector-1MB](https://blog.csdn.net/weixin_45250844/article/details/106161829)

[人脸检测--MTCNN从头到尾的详解](https://zhuanlan.zhihu.com/p/58825924)

## 注意

ncnn 与 aoce 调用二个不同的VkDevice,所以他们不能互相访问VkBuffer. [The fact that each program's VkDevice is distinct means you can't just straight up share VkImages](https://computergraphics.stackexchange.com/questions/6310/why-do-vulkan-extensions-need-to-be-enabled)
