#pragma once
#include <Layer/BaseLayer.hpp>

#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class AOCE_VULKAN_EXPORT VkResizeLayer : public VkLayer, public ReSizeLayer {
    AOCE_LAYER_QUERYINTERFACE(VkResizeLayer)
   private:
    /* data */
    ImageType imageType = ImageType::rgba8;

   public:
    VkResizeLayer();
    VkResizeLayer(ImageType imageType);
    ~VkResizeLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;

    virtual void onInitPipe() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce