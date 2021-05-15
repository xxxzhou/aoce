#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkPosterizeLayer : public VkLayer, public ITLayer<uint32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkPosterizeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPosterizeLayer(/* args */);
    ~VkPosterizeLayer();
};

class VkVignetteLayer : public VkLayer, public ITLayer<VignetteParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkVignetteLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkVignetteLayer(/* args */);
    ~VkVignetteLayer();
};

class VkCGAColorspaceLayer : public VkLayer {
   private:
    /* data */
   public:
    VkCGAColorspaceLayer(/* args */);
    virtual ~VkCGAColorspaceLayer();

   protected:
    virtual bool getSampled(int inIndex) override;
};

class VkCrosshatchLayer : public VkLayer, public ITLayer<CrosshatchParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkCrosshatchLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkCrosshatchLayer(/* args */);
    virtual ~VkCrosshatchLayer();
};

// The strength of the embossing, from  0.0 to 4.0, with 1.0 as the normal level
class VkEmbossLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkEmbossLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkEmbossLayer(/* args */);
    virtual ~VkEmbossLayer();
};

// 半径 1-32,默认为5
class VkKuwaharaLayer : public VkLayer, public ITLayer<uint32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkKuwaharaLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkKuwaharaLayer(/* args */);
    ~VkKuwaharaLayer();
};

}
}
}