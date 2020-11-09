#include <AoceManager.hpp>
#include <Media/MediaPlayer.hpp>
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

static MediaPlayer *player = nullptr;
// test rtmp
static std::string uri = "rtmp://202.69.69.180:443/webcast/bshdlive-pc";

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
                         std::string msg) override {
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
            int32_t size = getVideoFrame(frame, nullptr);
            bFill = size == 0;
            if (!bFill) {
                data.resize(size);
            }
        }
        if (bFill) {
            inputLayer->inputCpuData(frame.data[0], 0);
        } else {
            getVideoFrame(frame, data.data());
            inputLayer->inputCpuData(data.data(), 0);
        }
        vkGraph->run();
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
    outputLayer->outGpuTex(outTex);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

static TestMediaPlay *testPlay = nullptr;

#if _WIN32
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
#else __ANDROID__
void android_main(struct android_app *app)
#endif
{
    loadAoce();
#if __ANDROID__
    app_dummy();
    // 如果是要用nativewindow,请使用这个
    AoceManager::Get().initAndroid(app);
    AoceManager::Get().attachThread();
#endif

    // 生成一张执行图
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto *layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    // 输出GPU数据
    outputLayer->updateParamet({false, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);

    player = AoceManager::Get().getMediaPlayer(MediaPlayType::ffmpeg);
    // 这段在android需要窗口创建后才能用
    std::function<void()> winINit([&]() {
        while (!window->windowCreate()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        testPlay = new TestMediaPlay();
        player->setDataSource(uri);
        player->setObserver(testPlay);
        player->prepare(true);
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