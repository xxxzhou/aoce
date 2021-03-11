#if WIN32

#include "VkExtraBaseView.hpp"
#include "aoce_vulkan_extra/VkExtraExport.hpp"

static VkExtraBaseView* view = nullptr;

// box模糊
static ITLayer<KernelSizeParamet>* boxFilterLayer = nullptr;
static ITLayer<GaussianBlurParamet>* gaussianLayer = nullptr;
static ITLayer<ChromKeyParamet>* chromKeyLayer = nullptr;
static ITLayer<AdaptiveThresholdParamet>* adaptiveLayer = nullptr;
static BaseLayer* alphaShowLayer = nullptr;

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

    gaussianLayer = createGaussianBlurLayer();
    gaussianLayer->updateParamet({10, 5.0f});

    chromKeyLayer = createChromKeyLayer();
    ChromKeyParamet keyParamet = {};
    keyParamet.despillCuttofMax = 0.8f;
    keyParamet.despillExponent = 0.5f;
    keyParamet.ambientScale = 1.0f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.1f, 0.6f, 0.1f};
    keyParamet.alphaCutoffMin = 0.001f;
    chromKeyLayer->updateParamet(keyParamet);

    adaptiveLayer = createAdaptiveThresholdLayer();
    AdaptiveThresholdParamet adaParamet = {};
    adaParamet.boxSize = 10;
    adaParamet.offset = 0.01f;
    adaptiveLayer->updateParamet(adaParamet);

    alphaShowLayer = createAlphaShowLayer();

    // 查看自适应阈值化效果
    // view->initGraph(adaptiveLayer, hInstance, alphaShowLayer);
    // 查看高斯模糊效果
    view->initGraph(gaussianLayer, hInstance);
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