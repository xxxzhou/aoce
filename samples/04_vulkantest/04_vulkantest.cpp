#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

using namespace aoce;
using namespace aoce::vulkan;

static int index = 0;
static int formatIndex = 0;
static std::unique_ptr<VulkanContext> context = {};
static std::unique_ptr<VulkanWindow> window = nullptr;
static PipeGraph* vkGraph;
static InputLayer* inputLayer;
static OutputLayer* outputLayer;
static YUV2RGBALayer* yuv2rgbLayer;

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
    // context->BlitFillImage(cmd, csDestTex.get(), winImage, window->width,
    //                        window->height);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void onPreDraw() {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &context->computerCmd;

    vkQueueSubmit(context->graphicsQueue, 1, &submitInfo, nullptr);
}

#if _WIN32
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
#else __ANDROID__
void android_main(struct android_app* app)
#endif
{
    loadAoce();
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    if (deviceList.size() > 1) {
        index = 1;
    }
    VideoDevicePtr video = deviceList[index];
    std::wstring name((wchar_t*)video->getName().data());
    std::wstring id((wchar_t*)video->getId().data());
    std::wcout << "name: " << name << std::endl;
    std::wcout << "id: " << id << std::endl;
    auto& formats = video->getFormats();
    std::wcout << "formats count: " << formats.size() << std::endl;
    for (const auto& vf : formats) {
        std::cout << "index:" << vf.index << " width: " << vf.width
                  << " hight:" << vf.height << " fps:" << vf.fps
                  << " format:" << to_string(vf.videoType) << std::endl;
    }
    if (formats.size() > 355) {
        formatIndex = 355;
    }
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    video->setVideoFrameHandle(onDrawFrame);

    // 生成一张执行图
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    yuv2rgbLayer->updateParamet({VideoType::nv12});
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
    // 设定输入格式
    inputLayer->setImage(selectFormat);

    // window窗口
    context = std::make_unique<VulkanContext>();
    context->InitContext();
    window = std::make_unique<VulkanWindow>(context.get());
    window->InitWindow(hInstance, 1280, 720, "vulkan test");
    window->CreateSwipChain(context->logicalDevice.device, onPreCommand);
    window->Run(onPreDraw);

    unloadAoce();
}
