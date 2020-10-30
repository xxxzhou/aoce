#include <AoceManager.hpp>
#include <Live/LiveCallback.hpp>
#include <Live/LiveRoom.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

#ifdef __ANDROID__

#include <android/native_activity.h>
#include <android_native_app_glue.h>

#include "../../code/aoce/Aoce.h"
#include "errno.h"

#endif

using namespace aoce;
using namespace aoce::vulkan;

static std::unique_ptr<VulkanWindow> window = nullptr;
static PipeGraph *vkGraph;
static InputLayer *inputLayer;
static OutputLayer *outputLayer;
static YUV2RGBALayer *yuv2rgbLayer;
static VideoFormat format = {};

static GpuType gpuType = GpuType::vulkan;

// YUV数据是否紧密
static bool bFill = true;
static std::vector<uint8_t> data;

class TestLive : public LiveCallback {
   private:
    LiveRoom *room = nullptr;

   public:
    TestLive(LiveRoom *room) { this->room = room; }

    ~TestLive(){};

   public:
    // 网络发生的各种情况与处理码,如断网,网络情况不好等
    virtual void onEvent(int32_t operater, int32_t code, LogLevel level,
                         const std::string &msg) {
        std::string str;
        string_format(str, "code: ", code, " msg: ", msg);
        logMessage(level, str);
    };

    //
    virtual void onInitRoom(){};

    // loginRoom的网络应答
    virtual void onLoginRoom(bool bReConnect = false) {
        logMessage(LogLevel::info, "login success");
    };

    // 加入的房间人数变化
    virtual void onUserChange(int32_t userId, bool bAdd) {
        std::string str;
        string_format(str, (bAdd ? "add" : "remove"), "user: ", userId);
        logMessage(LogLevel::info, str);
        if (bAdd) {
            PullSetting setting = {};
            room->pullStream(userId, 0, setting);
        } else {
            room->stopPullStream(userId, 0);
        }
    };

    // 自己pushStream/stopPushStream推流回调,code不为0应该是出错了
    virtual void onStreamUpdate(int32_t index, bool bAdd, int32_t code){};

    // 别的用户pullStream/stopPullStream拉流回调
    virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd,
                                int32_t code) {
        std::string str;
        string_format(str, (bAdd ? "add" : "remove"), "stream ,user: ", userId);
        logMessage(LogLevel::info, str);
    };

    // 用户对应流的视频桢数据
    virtual void onVideoFrame(int32_t userId, int32_t index,
                              const VideoFrame &videoFrame) {
        if (format.width != videoFrame.width ||
            format.height != videoFrame.height) {
            format.width = videoFrame.width;
            format.height = videoFrame.height;
            format.videoType = videoFrame.videoType;
            inputLayer->setImage(format);
            yuv2rgbLayer->updateParamet({format.videoType});
            int32_t size = getVideoFrame(videoFrame, nullptr);
            bFill = size == 0;
            if (!bFill) {
                data.resize(size);
            }
        }
        if (bFill) {
            inputLayer->inputCpuData(videoFrame.data[0], 0);
        } else {
            getVideoFrame(videoFrame, data.data());
            inputLayer->inputCpuData(data.data(), 0);
        }
        vkGraph->run();
    };

    // 用户对应流的音频桢数据
    virtual void onAudioFrame(int32_t userId, int32_t index,
                              const AudioFrame &audioFrame){};

    // 推流的质量
    virtual void onPushQuality(int32_t index, int32_t quality, float fps,
                               float kbs){};

    // 拉流质量
    virtual void onPullQuality(int32_t userId, int32_t index, int32_t quality,
                               float fps, float kbs){};

    // 拿出房间
    virtual void onLogoutRoom(){};
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
    outputLayer->outGpuTex(outTex);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

static TestLive *live = nullptr;

#if _WIN32
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
#else __ANDROID__

void android_main(struct android_app *app)
#endif
{
    loadAoce();
#if __ANDROID__
    app_dummy();
    AoceManager::Get().initAndroid(app);
    AoceManager::Get().setJNIEnv(app->activity->env);
    AoceManager::Get().getJNIContext();
#endif

    // 生成一张执行图
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto *layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    outputLayer->updateParamet({false, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);

    // 这段在android需要窗口创建后才能用
    std::function<void()> winINit([&]() {
        while (!window->windowCreate()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        LiveRoom *room = AoceManager::Get().getLiveRoom(LiveType::agora);
        live = new TestLive(room);
        AgoraContext contex = {};
        contex.bLoopback = true;
        room->initRoom(&contex, live);
        room->loginRoom("123", 5, 0);
    });

    // 因执行图里随时重启,会导致相应资源重启,故运行时确定commandbuffer
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
#if _WIN32
    window->initWindow(hInstance, 1280, 720, "vulkan test");
    winINit();
#else  // __ANDROID__
    window->initWindow(winINit);
#endif
    window->run();
    unloadAoce();
}