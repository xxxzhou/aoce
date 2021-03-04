#if WIN32

#include "VkExtraBaseView.hpp"
#include "aoce_vulkan_extra/VkExtraExport.hpp"

static VkExtraBaseView* view = nullptr;

// box模糊
static ITLayer<BoxBlueParamet>* boxFilterLayer = nullptr;
static ITLayer<ChromKeyParamet>* chromKeyLayer = nullptr;
static ITLayer<AdaptiveThresholdParamet>* adaptiveLayer = nullptr;

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

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
    adaParamet.boxBlue = {4,4};
    adaptiveLayer->updateParamet(adaParamet);

    view->initGraph(boxFilterLayer, hInstance);
    view->openDevice();

    std::thread trd([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        view->enableLayer(false);
    });
    trd.detach();

    view->run();

    unloadAoce();
}

#endif