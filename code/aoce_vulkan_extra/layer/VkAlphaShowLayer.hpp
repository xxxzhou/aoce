#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkAlphaShowLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkAlphaShowLayer)
   private:
    /* data */

   public:
    VkAlphaShowLayer();
    virtual ~VkAlphaShowLayer();

   protected:
    virtual void onInitLayer() override;
};

class VkAlphaShow2Layer : public VkLayer {
    AOCE_LAYER_GETNAME(VkAlphaShow2Layer)
   public:
    VkAlphaShow2Layer();
    virtual ~VkAlphaShow2Layer();

   protected:
    virtual void onInitGraph() override;
};

class VkAlphaSeparateLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkAlphaSeparateLayer)
   public:
    VkAlphaSeparateLayer();
    virtual ~VkAlphaSeparateLayer();

   protected:
    virtual void onInitGraph() override;
};

class VkAlphaCombinLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkAlphaCombinLayer)
   public:
    VkAlphaCombinLayer();
    virtual ~VkAlphaCombinLayer();

   protected:
    virtual void onInitGraph() override;
};

class VkTwoShowLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkTwoShowLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkTwoShowLayer(bool bRow = false);
    virtual ~VkTwoShowLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce