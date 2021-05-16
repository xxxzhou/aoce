#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBulgeDistortionLayer : public VkLayer,
                               public ITLayer<DistortionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkBulgeDistortionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkBulgeDistortionLayer(/* args */);
    virtual ~VkBulgeDistortionLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkPinchDistortionLayer : public VkLayer,
                               public ITLayer<DistortionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkPinchDistortionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPinchDistortionLayer(/* args */);
    ~VkPinchDistortionLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkPixellatePositionLayer : public VkLayer,
                                 public ITLayer<SelectiveParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkPixellatePositionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPixellatePositionLayer(/* args */);
    ~VkPixellatePositionLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkPolarPixellateLayer : public VkLayer,
                              public ITLayer<PolarPixellateParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkPolarPixellateLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPolarPixellateLayer(/* args */);
    ~VkPolarPixellateLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkStrectchDistortionLayer : public VkLayer, public ITLayer<vec2> {
    AOCE_LAYER_QUERYINTERFACE(VkStrectchDistortionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkStrectchDistortionLayer(/* args */);
    ~VkStrectchDistortionLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkSwirlLayer : public VkLayer, public ITLayer<SwirlParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSwirlLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSwirlLayer(/* args */);
    ~VkSwirlLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce