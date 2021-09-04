#pragma once

#include "CNNHelper.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"
#include "aoce_vulkan/vulkan/VulkanBuffer.hpp"
#include "aoce_vulkan_extra/VkExtraExport.h"

using namespace aoce::vulkan;
using namespace aoce::vulkan::layer;

namespace aoce {

class INcnnInLayerObserver {
   public:
    INcnnInLayerObserver() = default;
    virtual ~INcnnInLayerObserver(){};

   public:
    virtual void onResult(VulkanBuffer* buffer,
                          const ImageFormat& imageFormat) = 0;
};

class DrawProperty : public virtual IDrawProperty {
   public:
    DrawProperty();
    virtual ~DrawProperty();

   protected:
    bool bDraw = true;
    int32_t radius = 3;
    vec4 color = {1.0f, 0.0f, 0.0f, 1.0f};

   public:
    virtual void setDraw(bool bDraw) override;
    virtual void setDraw(int32_t radius, const vec4 color) override;
};

class VkNcnnInLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkNcnnInLayer)
   protected:
    /* data */
    // 直接把CS BUFFER传输可能会导致问题
    std::unique_ptr<VulkanBuffer> outBuffer = nullptr;
    // 中转BUFFER
    std::unique_ptr<VulkanBuffer> outBufferX = nullptr;
    ImageType outImageType = ImageType::bgr8;
    // ncnn::VkBufferMemory nBuffer = {};
    INcnnInLayerObserver* observer = nullptr;
    NcnnInParamet paramet = {};

   public:
    VkNcnnInLayer();
    virtual ~VkNcnnInLayer();

    virtual void setObserver(INcnnInLayerObserver* observer,
                             ImageType outType = ImageType::bgr8);
    virtual void updateParamet(const NcnnInParamet& paramet);

   protected:
    virtual bool getSampled(int32_t inIndex) override;

    virtual void onInitGraph() override;
    virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override;
    virtual void onCommand() override;
    virtual bool onFrame() override;
};

class VkNcnnInCropLayer : public VkNcnnInLayer, public INcnnInCropLayer {
    AOCE_LAYER_QUERYINTERFACE(VkNcnnInCropLayer)
   public:
    VkNcnnInCropLayer(/* args */);
    virtual ~VkNcnnInCropLayer();

   protected:
    NcnnInCropParamet cropParamet = {};
    FaceBox faceBox = {};
    bool bFindBox = false;

   public:
    virtual void setObserver(INcnnInLayerObserver* observer,
                             ImageType outType = ImageType::bgr8) override;
    virtual void updateParamet(const NcnnInParamet& paramet) override;

    void getInFaceBox(FaceBox& faceBox);

   public:
    virtual void detectFaceBox(const FaceBox* boxs, int32_t lenght) override;
    virtual bool onFrame() override;
};

}  // namespace aoce