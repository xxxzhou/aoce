#if WIN32
#include <memory>
#include <vector>

#include "VkExtraBaseView.hpp"
#include "aoce_vulkan_extra/VkExtraExport.hpp"

static VkExtraBaseView* view = nullptr;

// box模糊
static ITLayer<KernelSizeParamet>* boxFilterLayer = nullptr;
static ITLayer<GaussianBlurParamet>* gaussianLayer = nullptr;
static ITLayer<ChromKeyParamet>* chromKeyLayer = nullptr;
static ITLayer<AdaptiveThresholdParamet>* adaptiveLayer = nullptr;
static ITLayer<GuidedParamet>* guidedLayer = nullptr;
static BaseLayer* convertLayer = nullptr;
static BaseLayer* alphaShowLayer = nullptr;
static ITLayer<ReSizeParamet>* resizeLayer = nullptr;
static ITLayer<KernelSizeParamet>* box1Layer = nullptr;
static ITLayer<GuidedMattingParamet>* guidedMattingLayer = nullptr;

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

    gaussianLayer = createGaussianBlurLayer();
    gaussianLayer->updateParamet({10, 5.0f});

    chromKeyLayer = createChromKeyLayer();
    ChromKeyParamet keyParamet = {};
    keyParamet.despillCuttofMax = 0.22f;
    keyParamet.despillExponent = 0.1f;
    keyParamet.ambientScale = 1.0f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.9f, 0.1f, 0.1f};
    keyParamet.alphaCutoffMax = 0.21f;
    keyParamet.alphaCutoffMin = 0.20f;
    chromKeyLayer->updateParamet(keyParamet);

    adaptiveLayer = createAdaptiveThresholdLayer();
    AdaptiveThresholdParamet adaParamet = {};
    adaParamet.boxSize = 10;
    adaParamet.offset = 0.01f;
    adaptiveLayer->updateParamet(adaParamet);

    alphaShowLayer = createAlphaShowLayer();
    convertLayer = createConvertImageLayer();
    resizeLayer = createResizeLayer(aoce::ImageType::rgbaf32);
    resizeLayer->updateParamet({false, 1920 / 8, 1080 / 8});
    box1Layer = createBoxFilterLayer(aoce::ImageType::rgbaf32);
    box1Layer->updateParamet({10, 10});
    guidedLayer = createGuidedLayer();

    guidedMattingLayer = createGuidedMattingLayer(chromKeyLayer->getLayer());

    // 查看自适应阈值化效果
    // view->initGraph(guidedLayer, hInstance, alphaShowLayer);
    // 查看高斯模糊效果
    std::vector<BaseLayer*> layers;
    layers.push_back(guidedMattingLayer->getLayer());
    view->initGraph(layers, hInstance);
    view->openDevice();
    // std::thread trd([&]() {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    //     view->enableLayer(false);
    // });
    // trd.detach();

    view->run();
    unloadAoce();
}

#endif