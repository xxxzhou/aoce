#include <jni.h>
#include <android/bitmap.h>
#include "Aoce.hpp"

using namespace aoce;

extern "C" {

ImageType getBitmapType(int32_t format){
    switch (format){
        case ANDROID_BITMAP_FORMAT_RGBA_8888:
            return ImageType::rgba8;
        case ANDROID_BITMAP_FORMAT_A_8:
            return ImageType::r8;
        default:
            return ImageType::other;
    }
}

JNIEXPORT jboolean JNICALL
Java_aoce_android_library_JNIHelper_loadBitmap(JNIEnv *env, jclass clazz, jlong inputLayerPtr,
                                               jobject bitmap) {
    IInputLayer *inputLayer = *(IInputLayer **) &inputLayerPtr;
    int result;
    // 获取源Bitmap相关信息：宽、高等
    AndroidBitmapInfo sourceInfo = {};
    result = AndroidBitmap_getInfo(env, bitmap, &sourceInfo);
    if (result < 0) {
        logMessage(LogLevel::warn, "android bitmap getInfo error");
        return false;
    }
    // 获取源Bitmap像素数据
    uint8_t *sourceData;
    result = AndroidBitmap_lockPixels(env, bitmap, (void **) &sourceData);
    if (result < 0) {
        logMessage(LogLevel::warn, "android bitmap lockPixels error");
        return false;
    }
    ImageType imageType = getBitmapType(sourceInfo.format);
    if (imageType == ImageType::other) {
        logMessage(LogLevel::warn, "android bitmap only support rgba8 or r8");
        return false;
    }
    // AndroidBitmapFormat
    ImageFormat imageFormat = {};
    imageFormat.width = sourceInfo.width;
    imageFormat.height = sourceInfo.height;
    imageFormat.imageType =imageType;
    inputLayer->inputCpuData(sourceData, imageFormat, true);
    AndroidBitmap_unlockPixels(env, bitmap);
    return true;
}

JNIEXPORT void JNICALL
Java_aoce_android_library_JNIHelper_setVideoFrameData(JNIEnv *env, jclass clazz, jlong video_frame,
                                                      jlong data) {
    VideoFrame *videoFrame = *(VideoFrame **) &video_frame;
    uint8_t *yuvdata = *(uint8_t **) &data;
    videoFrame->data[0] = yuvdata;
}

JNIEXPORT void JNICALL
Java_aoce_android_library_JNIHelper_setAgoraContext(JNIEnv *env, jclass clazz, jlong agora_context,
                                                    jobject context) {
    // TODO: implement setAgoraContext()
    AgoraContext *agoraContext = *(AgoraContext **) &agora_context;
    agoraContext->context =
            reinterpret_cast<void *>(env->NewGlobalRef(context));
}

}