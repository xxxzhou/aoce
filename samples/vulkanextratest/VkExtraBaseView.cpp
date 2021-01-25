#include "VkExtraBaseView.hpp"

using namespace std::placeholders;

VkExtraBaseView::VkExtraBaseView() { }

VkExtraBaseView::~VkExtraBaseView() { }

void VkExtraBaseView::initGraph(ILayer* layer, void* hinst) {
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    outputLayer->updateParamet({true, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    vkGraph->addNode(inputLayer)
        ->addNode(yuv2rgbLayer)
        ->addNode(layer)
        ->addNode(outputLayer);
    window = std::make_unique<VulkanWindow>(
        std::bind(&VkExtraBaseView::onPreCommand, this, _1), false);
#if _WIN32  
    window->initWindow((HINSTANCE)hinst, 1280, 720, "vulkan extra test");
#elif __ANDROID__
    window->initSurface((ANativeWindow*)hinst);
#endif
}

void VkExtraBaseView::openDevice() {
#if WIN32
    CameraType cameraType = CameraType::win_mf;
#elif __ANDROID__
    CameraType cameraType = CameraType::and_camera2;
#endif
    auto& deviceList =
        AoceManager::Get().getVideoManager(cameraType)->getDeviceList();
    VideoDevicePtr video = deviceList[index];
    auto& formats = video->getFormats();
    formatIndex = video->findFormatIndex(1920,1080);
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    video->setVideoFrameHandle(std::bind(&VkExtraBaseView::onFrame, this, _1));
    VideoType videoType = selectFormat.videoType;
    bool bYUV = false;
#if WIN32
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
#endif
    if (getYuvIndex(videoType) > 0) {
        yuv2rgbLayer->updateParamet({videoType});
    } else {
        bYUV = true;
    }
}

void VkExtraBaseView::onFrame(VideoFrame frame) {
    if(yuv2rgbLayer->getParamet().type != frame.videoType) {
        yuv2rgbLayer->updateParamet({frame.videoType});
    }
    inputLayer->inputCpuData(frame, 0);
    vkGraph->run();
#if __ANDROID
    if (window) {
        window->tick();
    }
#endif
}

void VkExtraBaseView::onPreCommand(uint32_t index) {
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


#if WIN32
void VkExtraBaseView::run() { window->run(); }
#endif