# 使用Swig转换C++到别的编程语言

设定aoce能分别与UE4/Unity3D/android demo对接,就这三来看,分别是C++/C#/java三种语言.

C++导出给别的语言使用,一般来说,分为二种,使用C风格,这种兼容性最好,但是问题是很多时候API明明是同一对象,使用C风格的就变成先创建/拿到类似句柄的东东,然后相关API第一个函数全是这个句柄做参数,用同一句柄表示他们是同一对象,有同样的上下文(各种状态这些).而第二种就是直接C++导出,别的用户能直接调用并有智能提示对应API,但是问题就在于各个C++编译器对于标准库实现的不同,如STL类库里的string,vector这些,就会导致别人使用你的库时,造成一些奇怪问题.

结合一些开源项目来看,使用abstract class与基本数据类型这种方式算是结合上面二种的优势.主要需要如下修改,一是如string这种,导出给外部使用的头文件里全换成const char*,而得到列表数据如vector,转化成接口getCount/getObject(index)这种.二是所有导出给外部使用的类,全部把接口抽离出来变成abstract class,而对应基类实现全放在不给外部用户使用的文件里.

``` c++
class IVideoManager {
   public:
    virtual ~IVideoManager(){};
    // 得到摄像头的个数
    virtual int32_t getDeviceCount(bool bUpdate = false) = 0;
    // 得到所有摄像头
    virtual void getDevices(IVideoDevice** videos, int32_t size,
                            int32_t start = 0) = 0;
    virtual IVideoDevice* getDevice(int32_t index) = 0;
};
```

然后是回调函数的实现,全面放弃C的函数指针或是std::function这种,转成observer abstract class接口这种实现,主要有三个好处,一是避免C这种要转换成无状态的静态函数,以及避免不同std::function编译器实现的坑,二是对应类型多回调的话,这种更方便,三是要转换到别的语言,Swig天然支持这种observer abstract class,不需要做更多处理.

``` c++
class IVideoDeviceObserver {
   public:
    virtual ~IVideoDeviceObserver(){};
    // 摄像机打开/关闭/丢失等事件
    virtual void onDeviceHandle(VideoHandleId id, int32_t codeId){};
    // 普通桢数据
    virtual void onVideoFrame(VideoFrame frame){};
    // 带深度的桢数据
    virtual void onDepthVideoFrame(VideoFrame colorFrame, VideoFrame depthFrame,
                                   void* alignParamt){};
};
```

最后就是返回上面abstract class对应框架内部具体实现对象以及一些帮助类,这种就用C风格的导出.

``` c++
extern "C" {

ACOE_EXPORT void setLogObserver(ILogObserver* observer);

// 检查模块aoce_cuda/aoce_vulkan/aoce_win_mf/aoce_ffmpeg是否加载成功
ACOE_EXPORT bool checkLoadModel(const char* modelName);

// 得到分配GPU管线的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT PipeGraphFactory* getPipeGraphFactory(const GpuType& gpuType);

// 得到分配基本层的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT LayerFactory* getLayerFactory(const GpuType& gpuType);

// 得到对应设备的管理对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT IVideoManager* getVideoManager(const CameraType& cameraType);

// 得到分配媒体播放器的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT MediaFactory* getMediaFactory(const MediaType& mediaType);

// 得到直播模块提供的管理对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT ILiveRoom* getLiveRoom(const LiveType& liveType);

// 得到对应计算层的参数元数据信息
ACOE_EXPORT ILMetadata* getLayerMetadata(const char* layerName);
}
```

在aoce里,整理出给外部用户的相关接口与API,可以看到在aoce里有多个.h的文件,这种文件就是放给外部用户,也是swig自动转换成别的语言需要处理的文件,当然在框架内部的各个模块里,还是直接使用C++导出带STL这种更方便.

## Swig注意事项

Swig的使用这里就不介绍了,就讲我在转换C#/Java时遇到一些需要注意的地方.

1 要使用回调observer类,需要打开swig的%module(directors = "1")这个,然后%feature("director") ILogObserver;声明相关类.

2 屏蔽相关的一些导出宏,如aoce模块里的ACOE_EXPORT,否则swig可能不能正常编译.

3 上面的getDevices(IVideoDevice** videos这种,本意是给C++申请空间后填充,但是对于别的语言来说,声明一个native的列表空间可能就很麻烦,然后swig默认也只能对二级指针生成SWIGTYPE_p_p这种无意义的对象,所以最好使用%ignore屏蔽掉.

4 如果类有中继承具体化模板的类,需要重新组织头文件,把这二个头分别放在不同头文件中,然后对对应二个%include之间使用%template指明具体化模板,否则生成的类可能丢失这个具体化模板的继承信息.

下面这个问题发生在android studio里,我本来直接使用swig把生成的java文件放入对应的package里,但是每次打开后,相应的package图标变成目录图标,还能正常运行,但是相关智能提示全部不能用,现解决方法是先复制到一个临时目录,然后从临时目录复制到这个package目录下.

然后通过cmake,每次更新编译都可以直接生成最新对应的java包,现用java就能完成以前C++编写的相关逻辑.

```java
public class AoceManager extends IVideoDeviceObserver {
    private IPipeGraph pipeGraph = null;
    private LayerFactory layerFactory = null;
    private IInputLayer inputLayer = null;
    private IOutputLayer outputLayer = null;
    private IYUVLayer yuv2RGBALayer = null;
    private ITransposeLayer transposeLayer = null;
    private IFlipLayer flipLayer = null;
    private IReSizeLayer reSizeLayer = null;
    private IBaseLayer extraLayer = null;
    private IVideoDevice videoDevice = null;
    private VideoFormat videoFormat = null;
    private GLOutGpuTex gpuTex = new GLOutGpuTex();

    public void initGraph(){
        // 生成VULKAN计算管线
        pipeGraph = AoceWrapper.getPipeGraphFactory(GpuType.vulkan).createGraph();
        layerFactory = AoceWrapper.getLayerFactory(GpuType.vulkan);
        // 输入层
        inputLayer = layerFactory.createInput();
        // 输出层
        outputLayer = layerFactory.createOutput();
        // YUV转RGBA
        yuv2RGBALayer = layerFactory.createYUV2RGBA();
        transposeLayer = layerFactory.createTranspose();
        flipLayer = layerFactory.createFlip();
        reSizeLayer = layerFactory.createSize();

        TransposeParamet tp = transposeLayer.getParamet();
        tp.setBFlipX(1);
        tp.setBFlipY(1);
        transposeLayer.updateParamet(tp);

        OutputParamet op = outputLayer.getParamet();
        op.setBGpu(1);
        op.setBCpu(0);
        outputLayer.updateParamet(op);
    }

    public void openCamera(){openCamera(false);}

    // 打开摄像机
    public void openCamera(boolean bFront){
        if(videoDevice != null && videoDevice.bOpen()){
            videoDevice.close();
        }
        // 底层用C++封装的NDK camera2 API,统一接口使用JAVA封装
        int deviceCount = AoceWrapper.getVideoManager(CameraType.and_camera2).getDeviceCount();
        for(int i=0;i<deviceCount;i++){
            videoDevice = AoceWrapper.getVideoManager(CameraType.and_camera2).getDevice(i);
            if(videoDevice.back() != bFront){
                break;
            }
        }
        // 找到1280x720的格式
        int formatIndex = videoDevice.findFormatIndex(1280,720);
        if(formatIndex < 0){
            formatIndex = 0;
        }
        videoDevice.setFormat(formatIndex);
        videoDevice.open();

        videoFormat = videoDevice.getSelectFormat();
        videoDevice.setObserver(this);
    }

    public void closeCamera(){
        if(videoDevice != null){
            videoDevice.close();
        }
    }

    public void initLayers(List<IBaseLayer> baseLayers,boolean bAutoIn){
        pipeGraph.clear();
        // 连接各层
        extraLayer = pipeGraph.addNode(inputLayer).addNode(yuv2RGBALayer);
        if(baseLayers != null) {
            for (IBaseLayer baseLayer : baseLayers) {
                extraLayer = extraLayer.addNode(baseLayer);
            }
            if (bAutoIn) {
                yuv2RGBALayer.getLayer().addLine(extraLayer, 0, 1);
            }
        }
        extraLayer.addNode(transposeLayer).addNode(outputLayer);
    }

    @Override
    public void onVideoFrame(VideoFrame frame){
        // 检查当前桢是否需要YUV转换
        if(AoceWrapper.getYuvIndex(frame.getVideoType()) < 0){
            yuv2RGBALayer.getLayer().setVisable(false);
        }else if(videoFormat != null){
            if(yuv2RGBALayer.getParamet().getType() != frame.getVideoType()){
                yuv2RGBALayer.getLayer().setVisable(true);
                YUVParamet yp = yuv2RGBALayer.getParamet();
                yp.setType(frame.getVideoType());
                yuv2RGBALayer.updateParamet(yp);
            }
        }
        inputLayer.inputCpuData(frame);
        pipeGraph.run();
    }

    public void showGL(int textureId, int width, int height){
        // 直接把VULKAN显示结果输出给OPENGL ES纹理
        gpuTex.setImage(textureId);
        gpuTex.setWidth(width);
        gpuTex.setHeight(height);
        outputLayer.outGLGpuTex(gpuTex);
    }
}

```

整个过程下来,使用CMake+Swig,就可以自动把相关C++更合适(GPU,图像处理,音视频)的方面转化成各种别的编程语言接口,方便别的编程语言更合适(界面UI)调用,修改更新逻辑也会自动更新完成,非常方便.
