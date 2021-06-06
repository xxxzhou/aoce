#include <AoceManager.hpp>
#include <iostream>
#include <media/MediaPlayer.hpp>
#include <memory>
#include <module/ModuleManager.hpp>
#include <string>
#include <thread>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

#ifdef __ANDROID__
#include <android/native_activity.h>
#include <android/native_window_jni.h>
#include <android_native_app_glue.h>

#include "../../code/aoce/Aoce.h"
#include "errno.h"

#endif

using namespace aoce;
using namespace aoce::vulkan;

static std::unique_ptr<VulkanWindow> window = nullptr;
static IPipeGraph *vkGraph;
static IInputLayer *inputLayer;
static IOutputLayer *outputLayer;
static IYUVLayer *yuv2rgbLayer;
static VideoFormat format = {};

static GpuType gpuType = GpuType::vulkan;

static IMediaPlayer *player = nullptr;
// test rtmp
// static std::string uri = "rtmp://202.69.69.180:443/webcast/bshdlive-pc";
static std::string uri = "rtmp://58.200.131.2:1935/livetv/hunantv";
// rtmp://58.200.131.2:1935/livetv/hunantv
// static std::string uri = "D://备份/tnmil3.flv";

class TestMediaPlay : public IMediaPlayerObserver {
   public:
    TestMediaPlay() {}

    virtual ~TestMediaPlay() override{};

   public:
    virtual void onPrepared() override {
        logMessage(LogLevel::info, "start");
        player->start();
    }

    virtual void onError(PlayStatus staus, int32_t code,
                         const char *msg) override {
        std::string mssg;
        string_format(mssg, "status: ", (int32_t)staus, " msg: ", msg);
        logMessage(LogLevel::info, mssg);
    };

    virtual void onVideoFrame(const VideoFrame &frame) override {
        std::string msg;
        string_format(msg, "time stamp: ", frame.timeStamp);
        logMessage(LogLevel::info, msg);
        if (format.width != frame.width || format.height != frame.height) {
            format.width = frame.width;
            format.height = frame.height;
            format.videoType = frame.videoType;
            inputLayer->setImage(format);
            yuv2rgbLayer->updateParamet({format.videoType});
        }
        inputLayer->inputCpuData(frame);
        vkGraph->run();
#if __ANDROID__
        if (window) {
            window->tick();
        }
#endif
    };

    virtual void onStop() override{};

    virtual void onComplate() override{};
};

void onPreCommand(uint32_t index) {
    VkImage winImage = window->images[index];
    VkCommandBuffer cmd = window->cmdBuffers[index];
    // 我们要把cs生成的图复制到正在渲染的图上,先改变渲染图的layout
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkOutGpuTex outTex = {};
    outTex.commandbuffer = cmd;
    outTex.image = winImage;
    outTex.width = window->width;
    outTex.height = window->height;
    outputLayer->outVkGpuTex(outTex);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

static TestMediaPlay *testPlay = nullptr;

#if _WIN32
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    // 生成一张执行图
    vkGraph = getPipeGraphFactory(gpuType)->createGraph();
    auto *layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->createInput();
    outputLayer = layerFactory->createOutput();
    // 输出GPU数据
    outputLayer->updateParamet({false, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);

    player = AoceManager::Get()
                 .getMediaFactory(MediaType::ffmpeg)
                 ->createPlay();
    // 因执行图里随时重启,会导致相应资源重启,故运行时确定commandbuffer
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initWindow(hInstance, 1280, 720, "vulkan test");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    testPlay = new TestMediaPlay();
    player->setDataSource(uri.c_str());
    player->setObserver(testPlay);
    player->prepare(true);

    window->run();
    unloadAoce();
}
#elif __ANDROID__

extern "C" {
JNIEXPORT void JNICALL Java_aoce_samples_mediaplayer_MainActivity_initEngine(
    JNIEnv *env, jobject thiz) {
    loadAoce();
    AndroidEnv andEnv = {};
    andEnv.env = env;
    andEnv.activity = thiz;
    AoceManager::Get().initAndroid(andEnv);
    // 生成一张执行图
    vkGraph = getPipeGraphFactory(gpuType)->createGraph();
    auto *layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->createInput();
    outputLayer = layerFactory->createOutput();
    // 输出GPU数据
    outputLayer->updateParamet({false, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
    player = AoceManager::Get()
                 .getMediaFactory(MediaType::ffmpeg)
                 ->createPlay();
    testPlay = new TestMediaPlay();
    player->setObserver(testPlay);
}

JNIEXPORT void JNICALL Java_aoce_samples_mediaplayer_MainActivity_vkInitSurface(
    JNIEnv *env, jobject thiz, jobject surface, jint width, jint height) {
    ANativeWindow *winSurf =
        surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initSurface(winSurf);
}

JNIEXPORT void JNICALL Java_aoce_samples_mediaplayer_MainActivity_glCopyTex(
    JNIEnv *env, jobject thiz, jint textureId, jint width, jint height) {
    GLOutGpuTex outGpuTex = {};
    outGpuTex.image =(int32_t)textureId;
    outGpuTex.width = width;
    outGpuTex.height = height;
    outputLayer->outGLGpuTex(outGpuTex);
}

JNIEXPORT void JNICALL Java_aoce_samples_mediaplayer_MainActivity_openUri(
    JNIEnv *env, jobject thiz, jstring uri) {
    const char *str = nullptr;
    jboolean bCopy = false;
    str = env->GetStringUTFChars(uri, &bCopy);
    if (str == NULL) {
        return;
    }
    // copy
    std::string ruri = str;
    env->ReleaseStringUTFChars(uri, str);
    player->setDataSource(ruri.c_str());
    player->prepare(true);
}
}
#endif
