#pragma once

#include "CNNHelper.hpp"
#include "VkNcnnInLayer.hpp"
#include "net.h"

namespace aoce {

#define VULKAN_OUTPUT 0

class VideoMatting : public IVideoMatting, public INcnnInLayerObserver {
   private:
    /* data */
    std::unique_ptr<ncnn::Net> net = nullptr;

    ncnn::VkAllocator* blobVkallocator = nullptr;
    ncnn::VkAllocator* stagingVkallocator = nullptr;

    VkNcnnInLayer* ncnnLayer = nullptr;
    VkNcnnUploadLayer* uploadLayer = nullptr;
    ImageFormat netFormet = {};
    bool bInitNet = false;

    float total = 0;
    int32_t totalIndex = 0;

#if VULKAN_OUTPUT
    std::unique_ptr<VkCommand> vkCmd = nullptr;
    ncnn::VkMat temp1;
    ncnn::VkMat temp2;
    ncnn::VkMat temp3;
    ncnn::VkMat temp4;
#else
    ncnn::Mat temp1;
    ncnn::Mat temp2;
    ncnn::Mat temp3;
    ncnn::Mat temp4;
#endif

   public:
    VideoMatting(/* args */);
    virtual ~VideoMatting();

   public:
    virtual bool initNet(IBaseLayer* ncnnInLayer,
                         IBaseLayer* ncnnUploadLayer) override;

   public:
    virtual void onResult(ncnn::VkMat& vkMat,
                          const ImageFormat& imageFormat) override;
};

}  // namespace aoce