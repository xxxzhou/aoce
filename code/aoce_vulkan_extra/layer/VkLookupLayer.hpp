#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VkBlendingModeLayer.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkInputLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkLookupLayer : public VkLayer, public LookupLayer {
    AOCE_LAYER_QUERYINTERFACE(VkLookupLayer)
   private:
    /* data */
    std::unique_ptr<VkInputLayer> lookupLayer = nullptr;

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

class VkSoftEleganceLayer : public GroupLayer, public SoftEleganceLayer {
    AOCE_LAYER_QUERYINTERFACE(VkSoftEleganceLayer)
   private:
    /* data */
    std::unique_ptr<VkLookupLayer> lookupLayer1 = nullptr;
    std::unique_ptr<VkLookupLayer> lookupLayer2 = nullptr;
    std::unique_ptr<VkAlphaBlendLayer> alphaBlendLayer = nullptr;
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer = nullptr;

   public:
    VkSoftEleganceLayer(/* args */);
    ~VkSoftEleganceLayer();

   public:
    virtual void loadLookUp1(uint8_t* data, int32_t size) override;
    virtual void loadLookUp2(uint8_t* data, int32_t size) override;

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce