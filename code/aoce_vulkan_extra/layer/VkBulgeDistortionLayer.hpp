#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBulgeDistortionLayer : public VkLayer,
                               public ITLayer<BulgeDistortionParamet> {
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

class VkGaussianBlurPositionLayer : public VkLayer,
                                    public ITLayer<BulrPositionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurPositionLayer)
   private:
    /* data */
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer = nullptr;

   public:
    VkGaussianBlurPositionLayer(/* args */);
    ~VkGaussianBlurPositionLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual bool getSampled(int32_t inIndex) override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

class VkGaussianBlurSelectiveLayer : public VkLayer,
                                    public ITLayer<BlurSelectiveParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurSelectiveLayer)
   private:
    /* data */
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer = nullptr;

   public:
    VkGaussianBlurSelectiveLayer(/* args */);
    ~VkGaussianBlurSelectiveLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual bool getSampled(int32_t inIndex) override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce