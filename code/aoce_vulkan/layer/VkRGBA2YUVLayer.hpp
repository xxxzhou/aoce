#pragma once
#include <layer/BaseLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
class VkRGBA2YUVLayer : public VkLayer, public IRGBA2YUVLayer {
    AOCE_LAYER_QUERYINTERFACE(VkRGBA2YUVLayer)
   private:
    /* data */
    std::unique_ptr<VulkanBuffer> kernelBuffer;
   public:
    VkRGBA2YUVLayer(/* args */);
    ~VkRGBA2YUVLayer();

   protected:   
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;

};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce