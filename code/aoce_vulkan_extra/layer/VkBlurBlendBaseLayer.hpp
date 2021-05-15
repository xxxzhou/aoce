#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBlurBlendBaseLayer : public VkLayer {
   protected:
    /* data */
    std::unique_ptr<VkGaussianBlurSLayer> blurLayer = nullptr;

   public:
    VkBlurBlendBaseLayer(/* args */);
    ~VkBlurBlendBaseLayer();

   protected:
    void baseParametChange(const GaussianBlurParamet& baseParamet);

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

class VkGaussianBlurPositionLayer : public VkBlurBlendBaseLayer,
                                    public ITLayer<BulrPositionParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurPositionLayer)
   public:
    VkGaussianBlurPositionLayer(/* args */);
    ~VkGaussianBlurPositionLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual bool getSampled(int32_t inIndex) override;
};

class VkGaussianBlurSelectiveLayer : public VkBlurBlendBaseLayer,
                                     public ITLayer<BlurSelectiveParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurSelectiveLayer)
   public:
    VkGaussianBlurSelectiveLayer(/* args */);
    ~VkGaussianBlurSelectiveLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual bool getSampled(int32_t inIndex) override;
};

class VkTiltShiftLayer : public VkBlurBlendBaseLayer,
                         public ITLayer<TiltShiftParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkTiltShiftLayer)
   public:
    VkTiltShiftLayer(/* args */);
    ~VkTiltShiftLayer();

   protected:
    void transformParamet();
    virtual void onUpdateParamet() override;
};

class VkUnsharpMaskLayer : public VkBlurBlendBaseLayer,
                           public ITLayer<UnsharpMaskParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkUnsharpMaskLayer)
   private:
    /* data */
   public:
    VkUnsharpMaskLayer(/* args */);
    ~VkUnsharpMaskLayer();

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce