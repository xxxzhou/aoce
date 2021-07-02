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
static IYUVLayer* r2yLayer = nullptr;
static IYUVLayer* y2rLayer = nullptr;
static ICaptureWindow* cw = nullptr;

static int32_t width = 1280;
static int32_t height = 720;

static bool bSecond = false;
static std::mutex mtx;

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
    r2yLayer = layerFactory->createRGBA2YUV();
    y2rLayer = layerFactory->createYUV2RGBA();

    inputLayer->updateParamet({false, true});
    outputLayer->updateParamet({false, true});
    r2yLayer->updateParamet({VideoType::yuv420P, 0});
    y2rLayer->updateParamet({VideoType::yuv420P, 0});
    resizeLayer->updateParamet({1, width, height});

    graph->addNode(inputLayer)
        ->addNode(resizeLayer)
        ->addNode(r2yLayer)
        ->addNode(y2rLayer)
        ->addNode(outputLayer);

    OutputLayerObserver outputObserver = {};
    outputLayer->setObserver(&outputObserver);

    cw = getWindowCapture(CaptureType::win_rt);
    IWindowManager* wm = getWindowManager(WindowType::win);
    int32_t winCount = wm->getWindowCount();
    IWindow* iw = nullptr;
    for (int32_t i = 0; i < winCount; i++) {
        std::string title = wm->getWindow(i)->getTitle();
        // Microsoft Visual/Visual Studio Code
        if (title.find("Visual Studio Code") != title.npos) {
            iw = wm->getWindow(i);
            break;
        }
    }
    cw->setObserver(new wincaptureObserver());
    // -----1-----
    // if (iw) {
    //     logMessage(LogLevel::info, iw->getTitle());
    //     cw->startCapture(iw, true);
    // }
    // -----2-----
    // logMessage(LogLevel::info, iw->getTitle());
    // cw->startCapture(iw, false);
    // std::thread pp([&]() {
    //     while (true) {
    //         cw->renderCapture();
    //         std::this_thread::sleep_for(std::chrono::milliseconds(30));
    //     }
    // });
    // pp.detach();
    // -----3-----
    // if (iw) {
    //     logMessage(LogLevel::info, iw->getTitle());
    //     cw->startCapture(iw, false);
    // }
    // -----4-----
    std::thread testTh([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        cw->stopCapture();
        bSecond = true;
        // graph->clearLines();
        // inputLayer->getLayer()->addLine(resizeLayer->getLayer());
        // resizeLayer->getLayer()->addLine(r2yLayer->getLayer());
        // r2yLayer->getLayer()->addLine(y2rLayer->getLayer());
        // y2rLayer->getLayer()->addLine(outputLayer->getLayer());

        graph->clear();
        graph->addNode(inputLayer)->addNode(resizeLayer)->addNode(outputLayer);

        cw->startCapture(iw, false);
        int i = 0;
        while (i++ < 100) {
            cw->renderCapture();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        
        graph->clear();
        graph->addNode(inputLayer)->addNode(resizeLayer)->addNode(outputLayer);
        while (true) {
            cw->renderCapture();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    });
    testTh.detach();
#if USE_DX11WINDOW
    window = std::make_unique<Dx11Window>();
    window->initWindow(hInstance, width, height, "dx11 caputure", onTick);
#else
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initWindow(hInstance, width, height, "vulkan caputure");
#endif
    window->run();
}