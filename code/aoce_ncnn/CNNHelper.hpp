#pragma once

#include "AoceNcnnExport.h"
#include "aoce_vulkan/VkCommand.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"
#include "aoce_vulkan/vulkan/VulkanBuffer.hpp"
#include "aoce_vulkan_extra/VkExtraExport.h"
#include "net.h"

using namespace aoce::vulkan;
using namespace aoce::vulkan::layer;

namespace aoce {

struct NcnnGlobeParamet {
    // aoce_vulkan是否与ncnn使用相同的VkDevice.
    bool bOneVkDevice = false;
    ncnn::VulkanDevice* vkDevice = nullptr;
    ncnn::VkAllocator* vkAllocator = nullptr;
};

const NcnnGlobeParamet& getNgParamet();

class INcnnInLayerObserver {
   public:
    INcnnInLayerObserver() = default;
    virtual ~INcnnInLayerObserver(){};

   public:
    virtual void onResult(ncnn::VkMat& vkMat,
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

ncnn::Mat getMat(uint8_t* data, const ImageFormat& inFormat,
                 const ImageFormat& outFormat);

int32_t getNetIndex(ncnn::Net* net, const char* blob_name);

void testVkMat(ncnn::VkMat& mat);

int32_t loadNet(ncnn::Net* net, const std::string& paramFile,
                const std::string& modelFile);

void copyBuffer(ncnn::Net* net, ncnn::VkMat& dstMat,
                aoce::vulkan::VulkanBuffer* srcbuffer);

}  // namespace aoce