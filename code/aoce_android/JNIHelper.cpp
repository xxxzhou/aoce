#include <jni.h>
#include <android/bitmap.h>
#include "Aoce.hpp"

using namespace aoce;

extern "C"
JNIEXPORT jboolean JNICALL
Java_aoce_android_library_JNIHelper_loadBitmap(JNIEnv *env, jclass clazz, jlong inputLayerPtr, jobject bitmap) {
    IInputLayer* inputLayer =  *(IInputLayer **)&inputLayerPtr;
    int result;
    // 获取源Bitmap相关信息：宽、高等
    AndroidBitmapInfo sourceInfo = {};
    result = AndroidBitmap_getInfo(env, bitmap, &sourceInfo);
    if (result < 0) {
        logMessage(LogLevel::warn,"android bitmap getInfo error");
        return false;
    }
    // 获取源Bitmap像素数据 这里用的是32位的int类型 argb每个8位
    uint8_t * sourceData;
    // 锁定像素的地址（不锁定的话地址可能会发生改变）
    result = AndroidBitmap_lockPixels(env, bitmap, (void**)& sourceData);
    if (result < 0) {
        logMessage(LogLevel::warn,"android bitmap lockPixels error");
        return false;
    }
    if(sourceInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888 && sourceInfo.format == ANDROID_BITMAP_FORMAT_A_8){
        logMessage(LogLevel::warn,"android bitmap only support rgba8 or r8");
        return false;
    }
    // AndroidBitmapFormat
    ImageFormat imageFormat = {};
    imageFormat.width = sourceInfo.width;
    imageFormat.height = sourceInfo.height;
    imageFormat.imageType = sourceInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ? ImageType::rgba8 : ImageType::r8;
    inputLayer->inputCpuData(sourceData,imageFormat, true);
    AndroidBitmap_unlockPixels(env, bitmap);
    return true;
}
