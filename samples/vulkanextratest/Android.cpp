#ifdef __ANDROID__

#include "VkExtraBaseView.hpp"

#include <android/native_activity.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android_native_app_glue.h>

#include "../../code/aoce/Aoce.h"
#include "errno.h"

static VkExtraBaseView* view = nullptr;

// box模糊
static ITLayer<FilterParamet>* boxFilterLayer = nullptr;
static ITLayer<ChromKeyParamet>* chromKeyLayer = nullptr;
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

    chromKeyLayer = createChromKeyLayer();    
    keyParamet.despillCuttofMax = 0.42f;
    keyParamet.despillExponent = 0.1f;
    keyParamet.ambientScale = 1.0f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.9f, 0.1f, 0.1f};
    keyParamet.alphaCutoffMin = 0.001f;
    chromKeyLayer->updateParamet(keyParamet);
    view->initGraph(chromKeyLayer, nullptr);
}


extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_glCopyTex(
    JNIEnv* env, jobject thiz, jint texture_id, jint width, jint height) {
    if(!view){
        return;
    }
    VkOutGpuTex outGpuTex = {};
    outGpuTex.image = texture_id;
    outGpuTex.width = width;
    outGpuTex.height = height;
    view->getOutputLayer()->outGLGpuTex(outGpuTex);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_openCamera(JNIEnv* env, jobject thiz,
                                                      jint index) {
    view->openDevice(index);
}

extern "C"
JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_enableLayer(JNIEnv *env, jobject thiz,
                                                           jboolean enable) {
    // TODO: implement enableLayer()
    view->enableLayer(enable);
}

extern "C"
JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_updateParamet(JNIEnv *env, jobject thiz,
                                                             jboolean green, jfloat luma,
                                                             jfloat min, jfloat max, jfloat scale) {
    // TODO: implement updateParamet()
    keyParamet.lumaMask = luma;
    keyParamet.alphaCutoffMin = min;
    keyParamet.alphaCutoffMin = max;
    keyParamet.alphaExponent = scale;
    aoce::vec3 color = {0.15f, 0.6f, 0.0f};
    if(!green){
        color = {0.15f,0.2f,0.8f};
    }
    keyParamet.chromaColor =color;
    chromKeyLayer->updateParamet(keyParamet);
}

#endif