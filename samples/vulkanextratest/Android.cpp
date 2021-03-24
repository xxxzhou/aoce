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
static ChromKeyParamet keyParamet = {};
static ITLayer<GuidedParamet>* guidedLayer = nullptr;

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

    std::vector<BaseLayer*> layers;
    layers.push_back(chromKeyLayer->getLayer());
    layers.push_back(guidedLayer->getLayer());
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
    jfloat scale, jfloat exponent,jfloat dscale) {
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