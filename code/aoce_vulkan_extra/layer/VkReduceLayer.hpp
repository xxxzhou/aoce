#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VKLuminanceLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkPreReduceLayer : public VkLayer {
   protected:
    ReduceOperate reduceType = ReduceOperate::sum;

   public:
    VkPreReduceLayer(ReduceOperate operate,
                     ImageType imageType = ImageType::rgba8);
    virtual ~VkPreReduceLayer();

   protected:
    ImageType imageType = ImageType::rgba8;

   protected:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
};

class VkReduceLayer : public VkLayer {
   protected:
    /* data */
    std::unique_ptr<VkPreReduceLayer> preLayer;
    ReduceOperate reduceType = ReduceOperate::sum;
    ImageType imageType = ImageType::rgba8;

   public:
    VkReduceLayer(ReduceOperate operate,
                  ImageType imageType = ImageType::rgba8);
    virtual ~VkReduceLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
};

class VkAverageLuminanceThresholdLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkAverageLuminanceThresholdLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    std::unique_ptr<VkLuminanceLayer> luminanceLayer;
    std::unique_ptr<VkReduceLayer> reduceLayer;

   public:
    VkAverageLuminanceThresholdLayer();
    virtual ~VkAverageLuminanceThresholdLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce