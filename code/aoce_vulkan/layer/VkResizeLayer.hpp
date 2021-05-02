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
    virtual bool getSampled(int inIndex) override;
    virtual bool sampledNearest(int32_t inIndex) override;
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

class AOCE_VULKAN_EXPORT VkSizeScaleLayer : public VkLayer,
                                            public ITLayer<SizeScaleParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSizeScaleLayer)
   private:
    /* data */
    ImageType imageType = ImageType::rgba8;

   public:
    VkSizeScaleLayer();
    VkSizeScaleLayer(ImageType imageType);
    ~VkSizeScaleLayer();

   protected:
    virtual bool getSampled(int inIndex) override;
    virtual bool sampledNearest(int32_t inIndex) override;
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce