#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkLuminanceLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkHistogramLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkHistogramLayer)
   private:
    /* data */
    int32_t channalCount = 1;

   public:
    VkHistogramLayer(bool signalChannal = true);
    ~VkHistogramLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
    virtual void onCommand() override;
};

class VkHistogramC4Layer : public VkLayer {
    AOCE_LAYER_GETNAME(VkHistogramC4Layer)
   private:
    /* data */
    std::unique_ptr<VkHistogramLayer> preLayer = nullptr;

   public:
    VkHistogramC4Layer(/* args */);
    ~VkHistogramC4Layer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
};

class VkHistogramLutLayer : public VkLayer, public ITLayer<int32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkHistogramLutLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkHistogramLutLayer(/* args */);
    ~VkHistogramLutLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

// 直方图均衡化
class VkEqualizeHistLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkEqualizeHistLayer)
   private:
    /* data */
    std::unique_ptr<VkHistogramLayer> histLayer = nullptr;
    std::unique_ptr<VkHistogramLutLayer> lutLayer = nullptr;
    std::unique_ptr<VkLuminanceLayer> lumLayer = nullptr;
    bool bSignal = true;

   public:
    VkEqualizeHistLayer(bool signalChannal = true);
    ~VkEqualizeHistLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce