#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkSphereRefractionLayer : public VkLayer,
                                public ITLayer<SphereRefractionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSphereRefractionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSphereRefractionLayer(/* args */);
    ~VkSphereRefractionLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkGlassSphereLayer : public VkLayer,
                           public ITLayer<SphereRefractionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGlassSphereLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkGlassSphereLayer(/* args */);
    ~VkGlassSphereLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce