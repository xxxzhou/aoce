#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 构造是传入false,更多的细节边.
class VkLaplacianLayer : public VkLayer {
   private:
    /* data */
    Mat3x3 mat = {};

   public:
    // small 中间-4,否则-8,边缘更深
    VkLaplacianLayer(bool small);
    ~VkLaplacianLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce