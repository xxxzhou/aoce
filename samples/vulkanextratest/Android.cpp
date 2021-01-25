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
    ChromKeyParamet keyParamet = {};
    keyParamet.ambientScale = 0.1f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.6f, 0.1f, 0.6f};
    keyParamet.alphaCutoffMin = 0.001f;
    chromKeyLayer->updateParamet(keyParamet);    
}

extern "C" JNIEXPORT void JNICALL 
Java_aoce_samples_vulkanextratest_MainActivity_vkInitSurface(
    JNIEnv* env, jobject thiz, jobject surface, jint width, jint height) {
    ANativeWindow* winSurf =
        surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
    view->initGraph(chromKeyLayer, winSurf);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_glCopyTex(
    JNIEnv* env, jobject thiz, jint texture_id, jint width, jint height) {
    VkOutGpuTex outGpuTex = {};
    outGpuTex.image = texture_id;
    outGpuTex.width = width;
    outGpuTex.height = height;
    view->getOutputLayer()->outGLGpuTex(outGpuTex);
}

extern "C" JNIEXPORT void JNICALL
Java_aoce_samples_vulkanextratest_MainActivity_openCamera(JNIEnv* env, jobject thiz,
                                                      jint index) {
    view->openDevice();
}

#endif