#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkBrightnessLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkBrightnessLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkBrightnessLayer(/* args */);
    virtual ~VkBrightnessLayer();
};

// Exposure ranges from -10.0 to 10.0, with 0.0 as the normal level
class VkExposureLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkExposureLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkExposureLayer(/* args */);
    virtual ~VkExposureLayer();
};

// Gamma ranges from 0.0 to 3.0, with 1.0 as the normal level
class VkGammaLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkGammaLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkGammaLayer(/* args */);
    virtual ~VkGammaLayer();
};

// 去雾
class VkHazeLayer : public VkLayer, public ITLayer<HazeParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkHazeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkHazeLayer(/* args */);
    virtual ~VkHazeLayer();
};

// 调整图像的阴影和高光
class VKHighlightShadowLayer : public VkLayer,
                               public ITLayer<HighlightShadowParamet> {
    AOCE_LAYER_QUERYINTERFACE(VKHighlightShadowLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VKHighlightShadowLayer(/* args */);
    virtual ~VKHighlightShadowLayer();
};

// 允许您使用颜色和强度独立地着色图像的阴影和高光
class VKHighlightShadowTintLayer : public VkLayer,
                                   public ITLayer<HighlightShadowTintParamet> {
    AOCE_LAYER_QUERYINTERFACE(VKHighlightShadowTintLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VKHighlightShadowTintLayer(/* args */);
    virtual ~VKHighlightShadowTintLayer();
};

// Saturation ranges from 0.0 (fully desaturated) to 2.0 (max saturation),
// with 1.0 as the normal level
class VkSaturationLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkSaturationLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSaturationLayer(/* args */);
    virtual ~VkSaturationLayer();
};

class VkMonochromeLayer : public VkLayer, public ITLayer<MonochromeParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkMonochromeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkMonochromeLayer(/* args */);
    virtual ~VkMonochromeLayer();
};

class VkOpacityLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkOpacityLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkOpacityLayer(/* args */);
    virtual ~VkOpacityLayer();
};

class VkPosterizeLayer : public VkLayer, public ITLayer<uint32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkPosterizeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPosterizeLayer(/* args */);
    ~VkPosterizeLayer();
};

class VkSharpenLayer : public VkLayer, public ITLayer<SharpenParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSharpenLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSharpenLayer(/* args */);
    ~VkSharpenLayer();
};

class VkSkinToneLayer : public VkLayer, public ITLayer<SkinToneParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkSkinToneLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSkinToneLayer(/* args */);
    ~VkSkinToneLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce