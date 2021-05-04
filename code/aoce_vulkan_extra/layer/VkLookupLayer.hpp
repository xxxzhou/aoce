#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkInputLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkLookupLayer : public VkLayer, public LookupLayer {
    AOCE_LAYER_QUERYINTERFACE(VkLookupLayer)
   private:
    /* data */
    std::unique_ptr<VkInputLayer> lookupLayer;

   public:
    VkLookupLayer(/* args */);
    virtual ~VkLookupLayer();

   public:
    virtual void loadLookUp(uint8_t* data, int32_t size) override;

   protected:
    virtual bool getSampled(int inIndex) override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

// class VkMosaicLayer : public VkLayer {
//    private:
//     /* data */
//    public:
//     VkMosaicLaye(/* args */);
//     ~VkMosaicLaye();
// };

// VkMosaicLayer::VkMosaicLaye(/* args */) {}

// VkMosaicLayer::~VkMosaicLaye() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce