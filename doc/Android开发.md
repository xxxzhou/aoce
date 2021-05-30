# Android开发

## 调试

jni调试不了,打开Project Structure里的Build Variants查找对应modules的jni debuggable是否为true.

android camera里的AImageReader_acquireNextImage用AImageReader_acquireLatestImage替换,可以避免AImage_delete出现异常.

[调试不能与Android Studio一起使用的C ++ /本机库模块](https://stackoverflow.com/questions/41822747/debugging-c-native-library-modules-not-working-with-android-studio-cmake-used)

原因似乎是，创建了lib的发行版，即使该应用程序是使用调试选项构建的，它也不支持调试。

在build.gradle里的android节点下添加一行: publishNonDefault  true

## Logcat

1. 过滤指定字符串，不让其显示的regex. ^(?!.*(字符串)).*$

## 注意

1 有返回值的一定要设置返回置,win里的msvc可能不在乎,android里的clang会在运行时给出SIGILL或SIGTRAP.

2 VkMotionDetectorLayer继承于IOutputLayerObserver,android里调试时会crash,指示SIGILL.

参考[what-causes-signal-sigill](https://stackoverflow.com/questions/7901867/what-causes-signal-sigill).

主要是因为使用的WIN上面的代码(只一函数main),而android多段函数,其中IOutputLayerObserver是一个堆栈对象,导致后面别的函数使用时已经消失.

3 android版本需要严格限制参数,如高斯模糊的glsl代码,里面逻辑限制核长32,window上参数超过也可执行,但是android下一般会导致vkQueueSubmit出现device lost错误.

4 导向滤波2021.05.29版本在android上不能运行.

经查在andorid中,卷积分离相关的类如VkSeparableLinearLayer(使用局部共享显存),其高度需要为16的整数倍才正常,应该线程组的分配方式有关.
