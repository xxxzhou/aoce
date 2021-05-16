#pragma once

#include "../VkExtraExport.h"
#include "VkLinearFilterLayer.hpp"
#include "VkLuminanceLayer.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 确定像素周围的局部亮度,如果像素低于该局部亮度,则将其变为黑色,如果高于该像素,则将其变为白色.
class VkAdaptiveThresholdLayer : public VkLayer,
                                 public ITLayer<AdaptiveThresholdParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkAdaptiveThresholdLayer)
   private:
    /* data */
    std::unique_ptr<VkLuminanceLayer> luminance;
    std::unique_ptr<VkBoxBlurSLayer> boxBlur;

   public:
    VkAdaptiveThresholdLayer(/* args */);
    virtual ~VkAdaptiveThresholdLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce