#pragma once

#include "aoce/AoceCore.h"
#include "aoce_vulkan_extra/VkExtraExport.h"

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

struct NcnnInParamet {
    int32_t outWidth = 0;
    int32_t outHeight = 0;
    vec4 mean = {0.0f, 0.0f, 0.0f, 0.0f};
    vec4 scale = {1.0f, 1.0f, 1.0f, 1.0f};
};

struct NcnnInCropParamet {
    NcnnInParamet ncnnIn = {};
    vec4 crop = {0.0f, 1.0f, 0.0f, 1.0f};
};

enum class FaceDetectorType { face, face_landmark };

class INcnnInCropLayer : public ILayer {
   public:
    INcnnInCropLayer() = default;
    virtual ~INcnnInCropLayer(){};

   public:
    virtual void detectFaceBox(const FaceBox* boxs, int32_t lenght) = 0;
};

// 面部识别结果通知
class IFaceObserver {
   public:
    IFaceObserver() = default;
    virtual ~IFaceObserver(){};

   public:
    virtual void onDetectorBox(const FaceBox* boxs, int32_t lenght) = 0;
};

// 面部关键点识别结果通知
class IFaceKeypointObserver {
   public:
    IFaceKeypointObserver() = default;
    virtual ~IFaceKeypointObserver(){};

   public:
    virtual void onDetectorBox(const vec2* points, int32_t lenght) = 0;
};

class IDrawProperty {
   public:
    IDrawProperty() = default;
    virtual ~IDrawProperty(){};

   public:
    virtual void setDraw(bool bDraw) = 0;
    virtual void setDraw(int32_t radius, const vec4 color) = 0;
};

class IFaceDetector : public virtual IDrawProperty {
   public:
    IFaceDetector() = default;
    virtual ~IFaceDetector(){};

   public:
    // 返回结果给CPU
    virtual void setObserver(IFaceObserver* observer) = 0;
    // 通知关联IFaceKeypointObserver的面部识别结果
    virtual void setFaceKeypointObserver(INcnnInCropLayer* cropLayer) = 0;
    // 其ncnnInLayer必需是createNcnnInLayer提供的运算层
    virtual bool initNet(IBaseLayer* ncnnInLayer,
                         IDrawRectLayer* drawLayer) = 0;
};

class IFaceKeypointDetector : public virtual IDrawProperty {
   public:
    IFaceKeypointDetector() = default;
    virtual ~IFaceKeypointDetector(){};

   public:
    // 返回结果给CPU
    virtual void setObserver(IFaceKeypointObserver* observer) = 0;
    virtual bool initNet(INcnnInCropLayer* ncnnInLayer,
                         IDrawPointsLayer* drawLayer) = 0;
};

class IVideoMatting {
   public:
    IVideoMatting() = default;
    virtual ~IVideoMatting(){};

   public:
    virtual bool initNet(IBaseLayer* ncnnInLayer,
                         IBaseLayer* ncnnUploadLayer) = 0;
};

extern "C" {

AOCE_NCNN_EXPORT IFaceDetector* createFaceDetector();

AOCE_NCNN_EXPORT IFaceKeypointDetector* createFaceKeypointDetector();

AOCE_NCNN_EXPORT IBaseLayer* createNcnnInLayer();

AOCE_NCNN_EXPORT INcnnInCropLayer* createNcnnInCropLayer();

AOCE_NCNN_EXPORT IVideoMatting* createVideoMatting();

AOCE_NCNN_EXPORT IBaseLayer* createNcnnUploadLayer();
}

}  // namespace aoce