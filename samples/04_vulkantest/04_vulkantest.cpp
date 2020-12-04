#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

// #define TEST_FIX_DATA

#ifdef __ANDROID__

#include <android/native_activity.h>
#include <android_native_app_glue.h>

#include "../../code/aoce/Aoce.h"
#include "errno.h"

#define TEST_FIX_DATA
#endif

#if !defined(TEST_FIX_DATA)
#define TEST_VIDEO_SHOW 1
#endif

using namespace aoce;
using namespace aoce::vulkan;

static int index = 0;
static int formatIndex = 29;
static std::unique_ptr<VulkanWindow> window = nullptr;
static PipeGraph *vkGraph;
static InputLayer *inputLayer;
static OutputLayer *outputLayer;
static YUV2RGBALayer *yuv2rgbLayer;

static GpuType gpuType = GpuType::vulkan;

static void onDrawFrame(VideoFrame frame) {
    // std::cout << "time stamp:" << frame.timeStamp << std::endl;
    inputLayer->inputCpuData(frame.data[0]);
    vkGraph->run();
}

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

#if _WIN32
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
#else __ANDROID__
void android_main(struct android_app *app)
#endif
{
#if __ANDROID__
    AoceManager::Get().initAndroid(app);
#endif
    loadAoce();
#if TEST_VIDEO_SHOW
    auto &deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    VideoDevicePtr video = deviceList[index];
    video->setVideoFrameHandle(onDrawFrame);
    std::wstring name((wchar_t *)video->getName().data());
    std::wstring id((wchar_t *)video->getId().data());
    std::wcout << "name: " << name << std::endl;
    std::wcout << "id: " << id << std::endl;
    auto &formats = video->getFormats();
    std::wcout << "formats count: " << formats.size() << std::endl;
    for (const auto &vf : formats) {
        std::cout << "index:" << vf.index << " width: " << vf.width
                  << " hight:" << vf.height << " fps:" << vf.fps
                  << " format:" << to_string(vf.videoType) << std::endl;
    }
    if (formats.size() > 355) {
        formatIndex = 355;
    }
    video->setFormat(formatIndex);
    video->open();
    auto &selectFormat = video->getSelectFormat();
#else  // __ANDROID__
    VideoFormat selectFormat = {};
    selectFormat.width = 1920;
    selectFormat.height = 1080;
    selectFormat.videoType = VideoType::nv12;
    std::vector<uint8_t> tempData(selectFormat.width * selectFormat.height * 4,
                                  255);
    for (int i = 0; i < selectFormat.height; i++) {
        for (int j = 0; j < selectFormat.width; j++) {
            float value = (float)j / selectFormat.width;
            value = value * i / selectFormat.height;
            tempData[i * selectFormat.width + j] = (int)(value * 255);
        }
    }
#endif
    // 生成一张执行图
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    outputLayer->updateParamet({true, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    VideoType videoType = selectFormat.videoType;
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
    if (getYuvIndex(videoType) > 0) {
        yuv2rgbLayer->updateParamet({videoType});
    }
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
    // 设定输入格式
    inputLayer->setImage(selectFormat);
#if !TEST_VIDEO_SHOW
    inputLayer->inputCpuData(tempData.data());
    vkGraph->run();
#endif
    // 因执行图里随时重启,会导致相应资源重启,故运行时确定commandbuffer
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
#if _WIN32
    window->initWindow(hInstance, 1280, 720, "vulkan test");
#else
    window->initWindow();
#endif
    window->run();

#if TEST_VIDEO_SHOW
    video.reset();
#endif
    unloadAoce();
}
