# aoce_android 针对android系统封装aoce接口

## android camera 部分

主要参考

[Using Android Native Camera API (CPU & GPU Processing)](https://www.sisik.eu/blog/android/ndk/camera)

[Native Camera API with OpenGL](https://github.com/sixo/native-camera/blob/master/app/src/main/cpp/native-lib.cpp)

[NdkCamera Sample](https://github.com/android/ndk-samples/tree/master/camera)

1. AIMAGE_FORMAT_YUV_420_888 可能是YUV420P,也可能是NV12,需要在AImageReader_ImageListener里拿到image通过AImage_getPlanePixelStride里的UV的plan是否为1来判断是否为YUV420P,或者看data[u]-data[y]=1来看是否为NV12.

2. AImageReader_new里的maxImages比较重要,简单理解为预先申请几张图,这个值越大,显示越平滑。
AImageReader_new如果不开线程,则图像处理加到这个线程里,导致读取图像变慢。打开线程处理,
我用的Redmi K10 pro,可以读4000*3000,在AImageReader_ImageListener回调不做特殊处理,如下错误。
首先是Unable to acquire a lockedBuffer, very likely client tries to lock more than.
可以看到,运行四次后报的,就是我设的maxImages,通过比对代码逻辑,应该是AImageReader_new读四次后,我还没处理完一桢,没有AImage_delete,也就读不了数据了.
然后检查 AImageReader_acquireNextImage 这个状态,不对不读,然后引发读取不可用内存问题,分析应该是处理数据的乱序线程AImage_delete可能释放别的处理线程上的image,然后处理图像线程上加上lock_guard(mutex),不会引发问题,但是会导致每maxImages卡一下,可以理解,读的线程快,处理的慢,后面想了下,直接让thread.join,图片读取很大时慢(比不开线程要快很多,4000*3000快二倍多,平均45ms),但是平滑的,暂时先这样,后面看能不能直接拿AImage的harderbuffer去处理,让处理速度追上读取速度。
