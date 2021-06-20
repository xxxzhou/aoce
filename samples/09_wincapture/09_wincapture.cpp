#include <AoceManager.hpp>
#include <media/MediaPlayer.hpp>
#include <module/ModuleManager.hpp>
#include <thread>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

#include "aoce/fixgraph/VideoView.hpp"
#include "aoce_win/dx11/Dx11Window.hpp"

using namespace aoce;
using namespace aoce::win;
using namespace aoce::vulkan;

#define USE_DX11WINDOW 1

#if USE_DX11WINDOW
static std::unique_ptr<Dx11Window> window = nullptr;
static GpuType gpuType = GpuType::vulkan;
#else
static std::unique_ptr<VulkanWindow> window = nullptr;
static GpuType gpuType = GpuType::vulkan;
#endif

static void* dx11Device = nullptr;
static void* dx11Tex = nullptr;

static IPipeGraph* graph = nullptr;
static IInputLayer* inputLayer = nullptr;
static IReSizeLayer* resizeLayer = nullptr;
static IOutputLayer* outputLayer = nullptr;
static ICaptureWindow* cw = nullptr;

#if USE_DX11WINDOW
static void onTick(void* dx11, void* tex) {
    dx11Device = dx11;
    dx11Tex = tex;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    outputLayer->outDx11GpuTex(dx11, dx11Tex);
}
#else
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
#endif

class OutputLayerObserver : public IOutputLayerObserver {
    virtual void onImageProcess(uint8_t* data, const ImageFormat& format,
                                int32_t outIndex) final {}
};

class wincaptureObserver : public ICaptureObserver {
   private:
    /* data */
   public:
    wincaptureObserver(/* args */){};
    virtual ~wincaptureObserver(){};

    virtual void onEvent(CaptureEventId eventId, LogLevel level,
                         const char* msg) override {
        logMessage(level, msg);
    };
    virtual void onCapture(const VideoFormat& videoFormat, void* device,
                           void* texture) override {
        inputLayer->setImage(videoFormat);
        inputLayer->inputGpuData(device, texture);
        graph->run();
    };
};

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();
    graph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->createInput();
    resizeLayer = layerFactory->createSize();
    outputLayer = layerFactory->createOutput();

    inputLayer->updateParamet({false, true});
    outputLayer->updateParamet({false, true});
    resizeLayer->updateParamet({1, 1280, 720});

    graph->addNode(inputLayer)->addNode(resizeLayer)->addNode(outputLayer);

    OutputLayerObserver outputObserver = {};
    outputLayer->setObserver(&outputObserver);

    cw = getWindowCapture(CaptureType::win_bitblt);
    IWindowManager* wm = getWindowManager(WindowType::win);
    int32_t winCount = wm->getWindowCount();
    IWindow* iw = nullptr;
    for (int32_t i = 0; i < winCount; i++) {
        std::string title = wm->getWindow(i)->getTitle();
        if (title.find("Microsoft Visual") != title.npos) {
            iw = wm->getWindow(i);
            break;
        }
    }
    if (iw) {
        logMessage(LogLevel::info, iw->getTitle());
        cw->setObserver(new wincaptureObserver());
        cw->startCapture(iw);
    }
#if USE_DX11WINDOW
    window = std::make_unique<Dx11Window>();
    window->initWindow(hInstance, 1280, 720, "dx11 caputure", onTick);
#else
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initWindow(hInstance, 1280, 720, "vulkan caputure");
#endif
    window->run();
}