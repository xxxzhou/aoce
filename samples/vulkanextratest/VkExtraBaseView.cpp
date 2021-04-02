#include "VkExtraBaseView.hpp"

using namespace std::placeholders;

VkExtraBaseView::VkExtraBaseView() {}

VkExtraBaseView::~VkExtraBaseView() {}

void VkExtraBaseView::initGraph(ILayer* layer, void* hinst,
                                BaseLayer* nextLayer) {
    std::vector<BaseLayer*> layers;
    layers.push_back(layer->getLayer());
    if (nextLayer != nullptr) {
        layers.push_back(nextLayer);
    }
    initGraph(layers, hinst);
}

void VkExtraBaseView::initGraph(std::vector<BaseLayer*> layers, void* hinst,
                                bool bAutoIn) {
    this->bAutoIn = bAutoIn;
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    outputLayer->updateParamet({false, true});
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    transposeLayer = layerFactory->createTranspose();
    operateLayer = layerFactory->createTexOperate();
    resizeLayer = layerFactory->createSize();
    resizeLayer->updateParamet({1, 240, 120});
    yuvNode = vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer);
    layerNode = yuvNode;
    for (auto& layer : layers) {
        layerNode = layerNode->addNode(layer);
    }
    if (layerNode->getLayer()->getInCount() == 2) {
        yuvNode->addLine(layerNode, 0, 1);
    }
#if _WIN32
    TexOperateParamet texParamet = {};
    texParamet.operate.bFlipX = false;
    texParamet.operate.bFlipY = false;
    // texParamet.operate.gamma = 0.45f;
    operateLayer->updateParamet(texParamet);
    layerNode->addNode(operateLayer)->addNode(outputLayer);  //
#elif __ANDROID__
    TransposeParamet tranParamet = {};
    tranParamet.bFlipX = true;
    tranParamet.bFlipY = true;
    transposeLayer->updateParamet(tranParamet);
    layerNode->addNode(transposeLayer)->addNode(outputLayer);
#endif
    if (hinst) {
        window = std::make_unique<VulkanWindow>(
            std::bind(&VkExtraBaseView::onPreCommand, this, _1), false);
#if _WIN32
        window->initWindow((HINSTANCE)hinst, 1280, 720, "vulkan extra test");
#elif __ANDROID__
        window->initSurface((ANativeWindow*)hinst);
#endif
    }
}

void VkExtraBaseView::openDevice(int32_t id) {
#if WIN32
    CameraType cameraType = CameraType::win_mf;  // realsense win_mf
#elif __ANDROID__
    CameraType cameraType = CameraType::and_camera2;
#endif
    auto& deviceList =
        AoceManager::Get().getVideoManager(cameraType)->getDeviceList();
    if (video != nullptr) {
        video->close();
    }
    index = id;
    video = deviceList[index];
    auto& formats = video->getFormats();
#if WIN32
    formatIndex = video->findFormatIndex(1920, 1080);
#elif __ANDROID__
    formatIndex = video->findFormatIndex(1280, 720);
#endif
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    video->setVideoFrameHandle(std::bind(&VkExtraBaseView::onFrame, this, _1));
    VideoType videoType = selectFormat.videoType;
#if WIN32
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
#endif
}

void VkExtraBaseView::closeDevice() {
    if (video != nullptr) {
        video->close();
    }
}

void VkExtraBaseView::enableLayer(bool bEnable) {
    // if (layerNode) {
    //     layerNode->setVisable(bEnable);
    // }
}

void VkExtraBaseView::onFrame(VideoFrame frame) {
    if (getYuvIndex(frame.videoType) < 0) {
        if (layerNode->getLayer()->getInCount() == 2 && bAutoIn) {
            inputLayer->getLayerNode()->addLine(layerNode, 0, 1);
        }
        yuvNode->setVisable(false);
    } else if (yuv2rgbLayer->getParamet().type != frame.videoType) {
        yuvNode->setVisable(true);
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
    if (!window) {
        return;
    }
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