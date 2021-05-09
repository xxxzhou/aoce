#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkHistogramLayer : public VkLayer {
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

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce