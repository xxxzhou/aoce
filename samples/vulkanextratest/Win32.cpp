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
// 亮度平均阈值
static ITLayer<float>* averageLT = nullptr;
static ITLayer<BilateralParamet>* bilateralLayer = nullptr;
static ITLayer<BulgeDistortionParamet>* bdLayer = nullptr;
static ITLayer<CannyEdgeDetectionParamet>* cedLayer = nullptr;
static BaseLayer* cgaLayer = nullptr;
static ITLayer<int>* dilationLayer = nullptr;
static ITLayer<int>* erosionLayer = nullptr;

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

    gaussianLayer = createGaussianBlurLayer();
    gaussianLayer->updateParamet({10, 20.0f});

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
    hcdParamet.blueParamet = {5, 0.0f};

    hcdLayer->updateParamet(hcdParamet);

    boxFilterLayer1 = createBoxFilterLayer(ImageType::r8);
    boxFilterLayer1->updateParamet({5, 5});

    resizeLayer = createResizeLayer(ImageType::rgba8);
    resizeLayer->updateParamet({true, 1920 / 8, 1080 / 8});
    resizeLayer2 = createResizeLayer(ImageType::rgba8);
    resizeLayer2->updateParamet({true, 1920, 1080});

    convertLayer = createConvertImageLayer();

    averageLT = createAverageLuminanceThresholdLayer();

    bilateralLayer = createBilateralLayer();
    bilateralLayer->updateParamet({10, 10.0f, 10.0f});

    bdLayer = createBulgeDistortionLayer();
    BulgeDistortionParamet bdParamet = {};
    bdLayer->updateParamet(bdParamet);

    guidedLayer = createGuidedLayer();
    guidedLayer->updateParamet({20, 0.000001f});

    cedLayer = createCannyEdgeDetectionLayer();
    CannyEdgeDetectionParamet cedParamet = {};
    cedLayer->updateParamet(cedParamet);

    cgaLayer = createCGAColorspaceLayer();

    dilationLayer = createDilationLayer();
    // dilationLayer->updateParamet(10);

    erosionLayer = createErosionLayer();
    // erosionLayer->updateParamet(20);

    std::vector<BaseLayer*> layers;
    bool bAutoIn = false;
    // ---高斯模糊
    // layers.push_back(gaussianLayer->getLayer());
    // 检测resize效果
    // layers.push_back(resizeLayer->getLayer());
    // layers.push_back(resizeLayer2->getLayer());
    // ---查看自适应阈值化效果
    // layers.push_back(adaptiveLayer->getLayer());
    // layers.push_back(alphaShowLayer);
    // ---查看Harris 角点检测
    // bAutoIn = true;
    // layers.push_back(luminanceLayer);
    // layers.push_back(hcdLayer->getLayer());
    // layers.push_back(boxFilterLayer1->getLayer());
    // layers.push_back(alphaShow2Layer);
    // ---查看导向滤波效果
    // layers.push_back(chromKeyLayer->getLayer());
    // layers.push_back(guidedLayer->getLayer());
    // layers.push_back(alphaShowLayer);
    // ---平均亮度调整阈值
    // layers.push_back(averageLT->getLayer());
    // layers.push_back(alphaShowLayer);
    // ---双边滤波
    // layers.push_back(bilateralLayer->getLayer());
    // ---凸起失真，鱼眼效果
    // layers.push_back(bdLayer->getLayer());
    // canny边缘检测
    // layers.push_back(cedLayer->getLayer());
    // layers.push_back(alphaShowLayer);
    // ---CGAColorspace效果
    // layers.push_back(cgaLayer);
    // ---dilation/erosion
    layers.push_back(cedLayer->getLayer());
    layers.push_back(dilationLayer->getLayer());
    layers.push_back(erosionLayer->getLayer());
    layers.push_back(alphaShowLayer);

    view->initGraph(layers, hInstance, bAutoIn);
    view->openDevice();

    // std::thread trd([&]() {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    //     view->enableLayer(false);
    // });

    view->run();
    unloadAoce();
}

#endif