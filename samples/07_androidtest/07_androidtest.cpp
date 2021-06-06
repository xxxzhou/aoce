#include "aoce/fixgraph/VideoView.hpp"
#include <android/native_activity.h>
#include <android/native_window_jni.h>
#include <android_native_app_glue.h>

#include <AoceManager.hpp>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

using namespace aoce;
using namespace aoce::vulkan;

static int index = 0;
static int formatIndex = 0;
static VideoDevicePtr video = nullptr;
static std::unique_ptr<VulkanWindow> window = nullptr;
static std::unique_ptr<VideoView> viewGraph = nullptr;
static std::unique_ptr<class TestCameraObserver> cameraObserver = nullptr;

class TestCameraObserver : public IVideoDeviceObserver {
    virtual void onVideoFrame(VideoFrame frame) override {
        std::string msg;
        string_format(msg, "time stamp: ", getNowTimeStamp());
        logMessage(LogLevel::info, msg);
        viewGraph->runFrame(frame);
        if (window) {
            window->tick();
        }
    }
    virtual void onDepthVideoFrame(VideoFrame frame, VideoFrame depth,
                                   void* alignParamet) final{

    }
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
    viewGraph->getOutputLayer()->outVkGpuTex(outTex);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void android_main(struct android_app* app) {
    AoceManager::Get().initAndroid(app);
    loadAoce();

    auto& deviceList = AoceManager::Get()
                           .getVideoManager(CameraType::and_camera2)
                           ->getDeviceList();
    logMessage(LogLevel::info, "device count:" + deviceList.size());

    VideoDevicePtr video = deviceList[index];
    std::string name = video->getName();
    std::string id = video->getId();
    std::cout << "name: " << name << std::endl;
    std::cout << "id: " << id << std::endl;
    auto& formats = video->getFormats();
    std::cout << "formats count: " << formats.size() << std::endl;
    for (const auto& vf : formats) {
        std::string msg;
        string_format(msg, "index:", vf.index, " width: ", vf.width,
                      " hight:", vf.height, " fps:", vf.fps,
                      " format:", getVideoType(vf.videoType));
        logMessage(LogLevel::info, msg);
    }
    int32_t formatIndex = video->findFormatIndex(1920,1080,0);
    video->setFormat(formatIndex);
    video->open();
    auto selectFormat = video->getSelectFormat();
    TestCameraObserver cameraObserver = {};
    video->setObserver(&cameraObserver);

    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initWindow();
    window->run();

    unloadAoce();
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_androidtest_MainActivity_initEngine(JNIEnv* env,
                                                      jobject thiz) {
    // TODO: implement initEngine()
    AndroidEnv andEnv = {};
    andEnv.env = env;
    andEnv.activity = thiz;
    AoceManager::Get().initAndroid(andEnv);
    loadAoce();

    viewGraph = std::make_unique<VideoView>();
    cameraObserver = std::make_unique<TestCameraObserver>();
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_androidtest_MainActivity_glCopyTex(JNIEnv* env, jobject thiz,
                                                     jint texture_id,
                                                     jint width, jint height) {
    GLOutGpuTex outGpuTex = {};
    outGpuTex.image = texture_id;
    outGpuTex.width = width;
    outGpuTex.height = height;
    viewGraph->getOutputLayer()->outGLGpuTex(outGpuTex);
}
extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_androidtest_MainActivity_vkInitSurface(
    JNIEnv* env, jobject thiz, jobject surface, jint width, jint height) {
    ANativeWindow* winSurf =
        surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initSurface(winSurf);
}
extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_androidtest_MainActivity_openCamera(JNIEnv* env, jobject thiz,
                                                      jint index) {
    // TODO: implement openCamera()
    auto& deviceList = AoceManager::Get()
            .getVideoManager(CameraType::and_camera2)
            ->getDeviceList();
    logMessage(LogLevel::info, "device count:" + deviceList.size());
    if(video != nullptr){
        video->close();
    }
    video = deviceList[index];
    video->setObserver(cameraObserver.get());
    auto& formats = video->getFormats();
    for (const auto& vf : formats) {
        std::string msg;
        string_format(msg, "index:", vf.index, " width: ", vf.width,
                      " hight:", vf.height, " fps:", vf.fps,
                      " format:", getVideoType(vf.videoType));
        logMessage(LogLevel::info, msg);
    }
    // formatIndex = video->findFormatIndex(1920,1080,0);
    video->setFormat(formatIndex);
    auto selectFormat = video->getSelectFormat();
    VideoType videoType = selectFormat.videoType;
    viewGraph->getOutputLayer()->updateParamet({false,true});
    video->open();
}