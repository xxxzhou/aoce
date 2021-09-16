#pragma once

#include "CNNHelper.hpp"

namespace aoce {

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
    bool bFP16 = false;
    ncnn::VkAllocator* inAlloc = nullptr;
    ncnn::VkMat inVkMat = {};
    // vulkan管线是否与ncnn同一vkdevice,如果是,结果直接用vkMat
    bool bOneVkDevice = false;

   public:
    VkNcnnInLayer();
    virtual ~VkNcnnInLayer();

    virtual void setObserver(INcnnInLayerObserver* observer,
                             ImageType outType = ImageType::bgr8);
    void updateParamet(const NcnnInParamet& paramet, bool bFP16);
    virtual void onParametChange(bool bUpdateUBO);

   protected:
    virtual bool getSampled(int32_t inIndex) override;

    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
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
    virtual void onParametChange(bool bUpdateUBO);
    void getInFaceBox(FaceBox& faceBox);

   public:
    virtual void detectFaceBox(const FaceBox* boxs, int32_t lenght) override;
    virtual void onInitLayer() override;
    virtual bool onFrame() override;
};

class VkNcnnUploadLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkNcnnUploadLayer)
   protected:
    std::unique_ptr<VulkanBuffer> inBuffer = nullptr;
    ImageFormat imageFormat = {};
    bool bFP16 = false;

   public:
    VkNcnnUploadLayer();
    virtual ~VkNcnnUploadLayer();

    void setImageFormat(const ImageFormat& inFormat, bool bFP16);
    void uploadBuffer(const void* data);
    VulkanBuffer* getVkBuffer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
    virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override;
    virtual void onCommand() override;
};

}  // namespace aoce