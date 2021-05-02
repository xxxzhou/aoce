#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkColorMatrixLayer : public VkLayer, public ITLayer<ColorMatrixParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkColorMatrixLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkColorMatrixLayer(/* args */);
    virtual ~VkColorMatrixLayer();
};

class VkHSBLayer : public VkLayer, public HSBLayer {
    AOCE_LAYER_QUERYINTERFACE(VkHSBLayer)
   private:
    /* data */
    ColorMatrixParamet paramet = {};

   public:
    VkHSBLayer(/* args */);
    ~VkHSBLayer();

   private:
    void parametTransform();

   public:
    virtual void reset();
    virtual void rotateHue(const float& h);
    virtual void adjustSaturation(const float& h);
    virtual void adjustBrightness(const float& h);
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce