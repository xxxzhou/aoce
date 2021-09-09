#include <aoce_ncnn/AoceNcnnExport.h>

#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <string>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

using namespace aoce;
using namespace aoce::vulkan;

static int index = 0;
static int formatIndex = 0;
static IPipeGraph* vkGraph = nullptr;
static IInputLayer* inputLayer = nullptr;
static IOutputLayer* outputLayer = nullptr;
static IYUVLayer* yuv2rgbLayer = nullptr;

static IBaseLayer* vmInLayer = nullptr;
static IBaseLayer* uploadLayer = nullptr;
static IBaseLayer* alphaScaleLayer = nullptr;
static IVideoMatting* videoMatting = nullptr;
static IBaseLayer* alphaShowLayer = nullptr;

static std::unique_ptr<VulkanWindow> window = nullptr;

// cuda也可测试
static GpuType gpuType = GpuType::vulkan;

class TestCameraObserver : public IVideoDeviceObserver {
    virtual void onVideoFrame(VideoFrame frame) override {
        // std::cout << "time stamp:" << frame.timeStamp << std::endl;
        inputLayer->inputCpuData(frame);
        vkGraph->run();
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
    outputLayer->outVkGpuTex(outTex);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    videoMatting = createVideoMatting();
    vmInLayer = createNcnnInLayer();
    uploadLayer = createNcnnUploadLayer();
    alphaScaleLayer = createAlphaScaleCombinLayer();
    alphaShowLayer = createAlphaShowLayer();
    // video matting
    videoMatting->initNet(vmInLayer, uploadLayer);
    // 打开摄像机
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    VideoDevicePtr video = deviceList[index];
    auto& formats = video->getFormats();
    formatIndex = video->findFormatIndex(1280, 720);
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    TestCameraObserver cameraObserver = {};
    video->setObserver(&cameraObserver);

    // 生成一张执行图
    vkGraph = getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->createInput();
    outputLayer = layerFactory->createOutput();
    yuv2rgbLayer = layerFactory->createYUV2RGBA();

    outputLayer->updateParamet({false, true});

    VideoType videoType = selectFormat.videoType;
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
    if (getYuvIndex(videoType) > 0) {
        yuv2rgbLayer->updateParamet({videoType});
    }
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer);
    vkGraph->addNode(outputLayer);
    vkGraph->addNode(vmInLayer);
    vkGraph->addNode(uploadLayer);
    vkGraph->addNode(alphaScaleLayer)->addNode(alphaShowLayer);
    yuv2rgbLayer->getLayer()->addLine(vmInLayer);
    yuv2rgbLayer->getLayer()->addLine(alphaScaleLayer);
    uploadLayer->addLine(alphaScaleLayer, 0, 1);
    alphaShowLayer->addLine(outputLayer->getLayer());
    //yuv2rgbLayer->getLayer()->addLine(outputLayer->getLayer());
    // 设定输入格式
    inputLayer->setImage(selectFormat);

    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initWindow(hInstance, selectFormat.width, selectFormat.height,
                       "matting test");
    window->run();

    video.reset();
    unloadAoce();
}