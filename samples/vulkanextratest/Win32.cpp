#if WIN32

#include "VkExtraBaseView.hpp"
#include "aoce_vulkan_extra/VkExtraExport.hpp"

static VkExtraBaseView* view = nullptr;

// box模糊
static ITLayer<FilterParamet>* boxFilterLayer = nullptr;
static ITLayer<ChromKeyParamet>* chromKeyLayer = nullptr;

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

    chromKeyLayer = createChromKeyLayer();
    ChromKeyParamet keyParamet = {};
    keyParamet.ambientScale = 0.1f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.6f, 0.1f, 0.6f};
    keyParamet.alphaCutoffMin = 0.001f;
    chromKeyLayer->updateParamet(keyParamet);

    view->initGraph(chromKeyLayer, hInstance);
    view->openDevice();
    view->run();

    unloadAoce();
}

#endif