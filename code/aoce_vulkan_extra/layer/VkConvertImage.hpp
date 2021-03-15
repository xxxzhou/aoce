#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

enum class ConvertType {
    other = 0,
    rgba82rgba32f,
};

// RGBA8->RGBA32F
class VkConvertImage : public VkLayer {
   private:
    /* data */
   public:
    VkConvertImage(/* args */);
    ~VkConvertImage();
};

VkConvertImage::VkConvertImage(/* args */) {}

VkConvertImage::~VkConvertImage() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
