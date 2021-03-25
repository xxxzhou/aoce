#ifdef __ANDROID__

#include <android/native_activity.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android_native_app_glue.h>

#include "../../code/aoce/Aoce.h"
#include "VkExtraBaseView.hpp"
#include "errno.h"

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
static ITLayer<KernelSizeParamet>* box1Layer = nullptr;
static ITLayer<HarrisCornerDetectionParamet>* hcdLayer = nullptr;
static ITLayer<KernelSizeParamet>* boxFilterLayer1 = nullptr;

static ChromKeyParamet keyParamet = {};
extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_initEngine(JNIEnv* env,
                                                          jobject thiz) {
    AndroidEnv andEnv = {};
    andEnv.env = env;
    andEnv.activity = thiz;
    AoceManager::Get().initAndroid(andEnv);
    loadAoce();

    view = new VkExtraBaseView();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({4, 4});

    gaussianLayer = createGaussianBlurLayer();
    gaussianLayer->updateParamet({10, 5.0f});

    chromKeyLayer = createChromKeyLayer();
    keyParamet.alphaScale = 10.0f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.9f, 0.1f, 0.1f};
    keyParamet.despillScale = 10.0f;
    keyParamet.despillExponent = 0.1f;
    chromKeyLayer->updateParamet(keyParamet);

    guidedLayer = createGuidedLayer();
    guidedLayer->updateParamet({20, 0.00001f});

    hcdLayer = createHarrisCornerDetectionLayer();
    HarrisCornerDetectionParamet hcdParamet = {};
    hcdParamet.threshold = 0.4f;
    hcdParamet.harris = 0.04f;
    hcdParamet.edgeStrength = 1.0f;
    hcdLayer->updateParamet(hcdParamet);

    boxFilterLayer1 = createBoxFilterLayer(ImageType::r8);
    alphaShow2Layer = createAlphaShow2Layer();
    luminanceLayer = createLuminanceLayer();

    std::vector<BaseLayer*> layers;
    // // 导向滤波
    // layers.push_back(chromKeyLayer->getLayer());
    // layers.push_back(guidedLayer->getLayer());
    // 查看Harris 角点检测
    layers.push_back(luminanceLayer);
    layers.push_back(hcdLayer->getLayer());
    layers.push_back(boxFilterLayer1->getLayer());
    layers.push_back(alphaShow2Layer);
    view->initGraph(layers, nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_glCopyTex(
    JNIEnv* env, jobject thiz, jint texture_id, jint width, jint height) {
    if (!view) {
        return;
    }
    VkOutGpuTex outGpuTex = {};
    outGpuTex.image = texture_id;
    outGpuTex.width = width;
    outGpuTex.height = height;
    view->getOutputLayer()->outGLGpuTex(outGpuTex);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_openCamera(JNIEnv* env,
                                                          jobject thiz,
                                                          jint index) {
    view->openDevice(index);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_enableLayer(JNIEnv* env,
                                                           jobject thiz,
                                                           jboolean enable) {
    // TODO: implement enableLayer()
    view->enableLayer(enable);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_updateParamet(
    JNIEnv* env, jobject thiz, jboolean green, jfloat luma, jfloat min,
    jfloat scale, jfloat exponent, jfloat dscale) {
    // TODO: implement updateParamet()

    keyParamet.lumaMask = luma;
    keyParamet.alphaCutoffMin = min;
    keyParamet.alphaScale = scale * 10.f;
    keyParamet.alphaExponent = exponent;
    aoce::vec3 color = {0.15f, 0.6f, 0.0f};
    if (!green) {
        color = {0.15f, 0.2f, 0.8f};
    }
    keyParamet.chromaColor = color;
    keyParamet.despillScale = dscale;
    chromKeyLayer->updateParamet(keyParamet);
}

#endif