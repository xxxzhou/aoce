#pragma once

#include "../VkExtraExport.hpp"
#include "VkLowPassLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkCopyImageLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkCopyImageLayer)
   private:
    /* data */
   public:
    VkCopyImageLayer(/* args */);
    ~VkCopyImageLayer();
};

class VkPoissonBlendLayer : public VkLayer, public ITLayer<PoissonParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkPoissonBlendLayer)
   private:
    // 当前层会改变第二个输入的数据,所以内置一个层用来代替第二个输入
    std::unique_ptr<VkCopyImageLayer> copyLayer = nullptr;

   public:
    VkPoissonBlendLayer(/* args */);
    virtual ~VkPoissonBlendLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onCommand() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce