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

一般来说,网络都会使用fp16优化,输入CPU数据时会自动转化成对应的fp16GPU数据,而输入GPU数据时,需要自身来做处理.

输入Mat的通道数如果是4的倍数,那么Mat对应的VkMat通道数/4,elempack*4.详细请看VkCompute::record_upload,对应思路不明.

VkBlobAllocator没有CPU访问权限,要与内存交互请用VkStagingAllocator.

## ncnn Vulkan环境

1 VkInstance没有放出来,并且VkInstance对应ppEnabledExtensionNames在win32下没有包含VK_KHR_WIN32_SURFACE_EXTENSION_NAME,所以不能创建vulkan渲染窗口,需要补起相关才可以在文件VkNcnnModule.cpp里设置NCNN_WIN32_VULKAN_INSTANCE为1,否则要么使用setVulkanContext传入physical_device/vkdeivce,这样ncnn/aoce可以使用GPU显存数据直接输入输出,输入输出的fp32/fp16转换全在GPU上进行,但是窗口只能用opencv的,要么不使用setVulkanContext,ncnn/aoce只能通过CPU交换,可以使用vulkan渲染窗口.

2 android下的VkDevice的ppEnabledExtensionNames没包含VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,则不能直接使用绑定VkImage与opengl纹理,导致现不能把结果给opengl直接渲染.

现使用ncnn直接替换aoce的Vulkan环境还有一些问题,主要是对应的扩展不同,但是如果不替换,aoce计算的vulkan buffer我暂时还没找到好方法可以和ncnn输入/输出对接.现阶段是拿到ncnn的源码,修改如上二点,可分别在window/android使用vulkan/opengl显示与全vulkan对接输入输出.
