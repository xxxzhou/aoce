#pragma once

#include "../aoce/AoceCore.h"

#ifdef _WIN32
#if defined(AOCE_NCNN_EXTRA_EXPORT_DEFINE)
#define AOCE_NCNN_EXPORT __declspec(dllexport)
#else
#define AOCE_NCNN_EXPORT __declspec(dllimport)
#endif
#elif __ANDROID__
#if defined(AOCE_VULKAN_EXTRA_EXPORT_DEFINE)
#define AOCE_NCNN_EXPORT __attribute__((visibility("default")))
#else
#define AOCE_NCNN_EXPORT
#endif
#endif

namespace aoce {

struct Box {
    float cx;
    float cy;
    float sx;
    float sy;
};

struct FaceBox {
    float s;
    float x1;
    float x2;
    float y1;
    float y2;
    vec2 point[5];
};

struct NcnnInParamet{
    int32_t outWidth = 0;
    int32_t outHeight = 0;
    vec4 mean = {0.0f,0.0f,0.0f,0.0f};
    vec4 scale = {1.0f,1.0f,1.0f,1.0f};
};

enum class FaceDetectorType { face, face_landmark };

class IFaceDetectorObserver {
   public:
    IFaceDetectorObserver() = default;
    virtual ~IFaceDetectorObserver(){};

   public:
    virtual void onDetectorBox(const FaceBox* boxs, int32_t lenght) = 0;
};

class IFaceDetector {
   public:
    IFaceDetector() = default;
    virtual ~IFaceDetector(){};

   public:
    virtual void setObserver(IFaceDetectorObserver* observer) = 0;
    virtual bool initNet(FaceDetectorType detectorType) = 0;
    virtual void detect(uint8_t* data, const ImageFormat& inFormat) = 0;
};

extern "C" {
AOCE_NCNN_EXPORT IFaceDetector* createFaceDetector();
}

}  // namespace aoce