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
static BaseLayer* alphaShow2Layer = nullptr;
static BaseLayer* luminanceLayer = nullptr;
static ITLayer<ReSizeParamet>* resizeLayer = nullptr;
static ITLayer<ReSizeParamet>* resizeLayer2 = nullptr;
static ITLayer<KernelSizeParamet>* box1Layer = nullptr;
static ITLayer<HarrisCornerDetectionParamet>* hcdLayer = nullptr;
static ITLayer<KernelSizeParamet>* boxFilterLayer1 = nullptr;

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

    gaussianLayer = createGaussianBlurLayer();
    gaussianLayer->updateParamet({10, 5.0f});

    chromKeyLayer = createChromKeyLayer();
    ChromKeyParamet keyParamet = {};
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.alphaScale = 20.0f;
    keyParamet.alphaExponent = 0.1f;
    keyParamet.alphaCutoffMin = 0.2f;
    keyParamet.lumaMask = 2.0f;
    keyParamet.ambientColor = {0.1f, 0.1f, 0.9f};
    keyParamet.despillScale = 0.0f;
    keyParamet.despillExponent = 0.1f;
    keyParamet.ambientScale = 1.0f;
    chromKeyLayer->updateParamet(keyParamet);

    adaptiveLayer = createAdaptiveThresholdLayer();
    AdaptiveThresholdParamet adaParamet = {};
    adaParamet.boxSize = 10;
    adaParamet.offset = 0.01f;
    adaptiveLayer->updateParamet(adaParamet);

    alphaShowLayer = createAlphaShowLayer();
    alphaShow2Layer = createAlphaShow2Layer();
    luminanceLayer = createLuminanceLayer();

    hcdLayer = createHarrisCornerDetectionLayer();
    HarrisCornerDetectionParamet hcdParamet = {};
    hcdParamet.threshold = 0.1f;
    hcdParamet.harris = 0.04f;
    hcdParamet.edgeStrength = 1.0f;

    hcdLayer->updateParamet(hcdParamet);

    boxFilterLayer1 = createBoxFilterLayer(ImageType::r8);

    resizeLayer = createResizeLayer(ImageType::rgba8);
    resizeLayer->updateParamet({true, 1920 / 8, 1080 / 8});
    resizeLayer2 = createResizeLayer(ImageType::rgba8);
    resizeLayer2->updateParamet({true, 1920, 1080});

    convertLayer = createConvertImageLayer();

    guidedLayer = createGuidedLayer();
    guidedLayer->updateParamet({20, 0.000001f});
    std::vector<BaseLayer*> layers;
    // 检测resize效果    
    // layers.push_back(resizeLayer->getLayer());
    // layers.push_back(resizeLayer2->getLayer());    
    // 查看自适应阈值化效果
    // layers.push_back(adaptiveLayer->getLayer());
    // layers.push_back(alphaShowLayer);
    // 查看Harris 角点检测
    // layers.push_back(luminanceLayer);
    // layers.push_back(hcdLayer->getLayer());
    // layers.push_back(boxFilterLayer1->getLayer());
    // layers.push_back(alphaShow2Layer);
    // 查看导向滤波效果
    layers.push_back(chromKeyLayer->getLayer());
    layers.push_back(guidedLayer->getLayer());
    layers.push_back(alphaShowLayer);

    view->initGraph(layers, hInstance);
    view->openDevice();
    view->run();
    unloadAoce();
}

#endif