#pragma once

#include <memory>

#include "aoce_vulkan/layer/VkLayer.hpp"
#include "../VkExtraExport.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 边框默认使用REPLICATE模式
class VkLinearFilterLayer : public VkLayer,public ITLayer<FilterParamet> {    
    AOCE_LAYER_QUERYINTERFACE(VkLinearFilterLayer)
   protected:
    std::unique_ptr<VulkanBuffer> kernelBuffer;
   public:
    VkLinearFilterLayer(/* args */);
    ~VkLinearFilterLayer();

   protected:
    virtual void onUpdateParamet() override;

    virtual void onInitGraph() override;    
    virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
