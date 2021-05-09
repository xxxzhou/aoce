#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkPrewittEdgeDetectionLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkPrewittEdgeDetectionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPrewittEdgeDetectionLayer(/* args */);
    ~VkPrewittEdgeDetectionLayer();

   protected:
    virtual void onInitGraph() override;
};

class VkSobelEdgeDetectionLayer : public VkPrewittEdgeDetectionLayer {
    AOCE_LAYER_QUERYINTERFACE(VkPrewittEdgeDetectionLayer)
   private:
    /* data */
   public:
    VkSobelEdgeDetectionLayer(/* args */);
    ~VkSobelEdgeDetectionLayer();
};

class VkSketchLayer : public VkPrewittEdgeDetectionLayer {
    AOCE_LAYER_QUERYINTERFACE(VkSketchLayer)
   private:
    /* data */
   public:
    VkSketchLayer(/* args */);
    ~VkSketchLayer();
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce