#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// RGBA8->RGBA32F
class VkConvertImageLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkConvertImageLayer)
   private:
    /* data */
    ConvertType convert = ConvertType::rgba82rgba32f;

   public:
    VkConvertImageLayer(ConvertType convert = ConvertType::rgba82rgba32f);
    virtual ~VkConvertImageLayer();

   protected:
    virtual void onInitGraph() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
