#pragma once

#include "../VkExtraExport.hpp"
#include "VkLowPassLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkPoissonBlendLayer : public VkLayer {
   private:
    /* data */
    std::unique_ptr<VkSaveFrameLayer> saveLayer = nullptr;

   public:
    VkPoissonBlendLayer(/* args */);
    virtual ~VkPoissonBlendLayer();

//    protected:
//     virtual void onInitGraph() override;
//     virtual void onInitNode() override;
//     virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce