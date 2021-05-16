#pragma once

#include <memory>

#include "../VkExtraExport.h"
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

class VkHSBLayer : public VkLayer, public IHSBLayer {
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
    virtual void reset() override;
    virtual void rotateHue(const float& h) override;
    virtual void adjustSaturation(const float& h) override;
    virtual void adjustBrightness(const float& h) override;
};

class VkSepiaLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkSepiaLayer)
   private:
    /* data */
    ColorMatrixParamet mparamet = {};

   public:
    VkSepiaLayer(/* args */);
    ~VkSepiaLayer();

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce