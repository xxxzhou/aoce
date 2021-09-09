#pragma once

#include "CNNHelper.hpp"
#include "VkNcnnInLayer.hpp"
#include "net.h"

namespace aoce {

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

    ncnn::Mat temp1;
    ncnn::Mat temp2;
    ncnn::Mat temp3;
    ncnn::Mat temp4;

   public:
    VideoMatting(/* args */);
    virtual ~VideoMatting();

   public:
    virtual bool initNet(IBaseLayer* ncnnInLayer,
                         IBaseLayer* ncnnUploadLayer) override;

   public:
    virtual void onResult(VulkanBuffer* buffer,
                          const ImageFormat& imageFormat) override;
};

}  // namespace aoce