#pragma once

#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 边框默认使用REPLICATE模式
class VkSeparableLinearLayer : public VkLayer {    
   protected:
    /* data */
    int32_t rSizeX = 0;
    int32_t cSizeY = 0;

    std::unique_ptr<VulkanShader> shader2;

   public:
    VkSeparableLinearLayer(/* args */);
    ~VkSeparableLinearLayer();

   protected:
    virtual void onInitGraph();
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
