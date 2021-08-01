# Android开发

## 调试

jni调试不了,打开Project Structure里的Build Variants查找对应modules的jni debuggable是否为true.

android camera里的AImageReader_acquireNextImage用AImageReader_acquireLatestImage替换,可以避免AImage_delete出现异常.

[调试不能与Android Studio一起使用的C ++ /本机库模块](https://stackoverflow.com/questions/41822747/debugging-c-native-library-modules-not-working-with-android-studio-cmake-used)

原因似乎是，创建了lib的发行版，即使该应用程序是使用调试选项构建的，它也不支持调试。

在build.gradle里的android节点下添加一行: publishNonDefault  true

## Logcat

1. 过滤指定字符串，不让其显示的regex. ^(?!.*(字符串)).*$

2. adb logcat -v threadtime > log.txt(adb 无过滤加载所有log)

## 注意

1 有返回值的一定要设置返回置,win里的msvc可能不在乎,android里的clang会在运行时给出SIGILL或SIGTRAP.

2 VkMotionDetectorLayer继承于IOutputLayerObserver,android里调试时会crash,指示SIGILL.

参考[what-causes-signal-sigill](https://stackoverflow.com/questions/7901867/what-causes-signal-sigill).

主要是因为使用的WIN上面的代码(只一函数main),而android多段函数,其中IOutputLayerObserver是一个堆栈对象,导致后面别的函数使用时已经消失.

3 android版本需要严格限制参数,如高斯模糊的glsl代码,里面逻辑限制核长32,window上参数超过也可执行,但是android下一般会导致vkQueueSubmit出现device lost错误.

4 导向滤波2021.05.29版本在android上不能运行.

经查在andorid中,glsl代码filterRow需求其高度需要为16的整数倍才正常,应该线程组的分配方式有关.

解决方式:修改filterRow逻辑,在填充shared数据之前,不做线程组与大小的验证,shared填充满.不过奇怪的是,为什么不需要宽度是16的整数倍了,并且也不需要修改filterColumn的逻辑.

注意:以后如果有用到shared数据的逻辑glsl,一定在用到之前初始化/填充所有shared数据,避免这个问题.相应逻辑后期也需要全部修正,包含filterColumn.

5 使用cmake + swig生成的文件直接复制到模块包里,会导致android studio每次打开把相关包变成目录,使编缉器丢掉包名引用等智能提示出现错误.

可以多加一步,先复制到一个非android模块目录(最好是CMAKE的BINARY目录,否则android studio容易检测到重命),然后使用cmake里的file复制过去,可以解决.

6 对应Cmake输出文件在如下目录-project\.cxx\cmake\{build type}\{abi}\build_output.txt,如deubg/v8下对应(project\.cxx\cmake\debug\arm64-v8a\build_output.txt).
