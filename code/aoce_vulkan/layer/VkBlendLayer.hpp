#pragma once

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

struct VkBlendParamet {
    // 0
    float fx;
    float fy;
    // 0
    float centerX;
    float centerY;
    float width;
    float height;
    // 不透明
    float opacity;
};

class VkBlendLayer : public VkLayer, public BlendLayer {
    AOCE_LAYER_QUERYINTERFACE(VkBlendLayer)
   private:
    VkBlendParamet vkParamet = {};

   public:
    VkBlendLayer(/* args */);
    ~VkBlendLayer();

   private:
    void parametTransform();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;

    virtual void onInitPipe() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce