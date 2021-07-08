#pragma once

#include <memory>

#include "../VkExtraExport.h"
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

class VkRGBLayer : public VkLayer, public ITLayer<vec3> {
    AOCE_LAYER_QUERYINTERFACE(VkRGBLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkRGBLayer(/* args */);
    ~VkRGBLayer();
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

class VkSolarizeLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkSolarizeLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkSolarizeLayer(/* args */);
    ~VkSolarizeLayer();
};

// Modifies the saturation of desaturated colors, leaving saturated colors
// unmodified. Value -1 to 1 (-1 is minimum vibrance, 0 is no change, and 1 is
// maximum vibrance)
class VkVibranceLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkVibranceLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkVibranceLayer(/* args */);
    ~VkVibranceLayer();
};

class VkWhiteBalanceLayer : public VkLayer,
                            public ITLayer<WhiteBalanceParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkWhiteBalanceLayer)
   private:
    /* data */
   public:
    VkWhiteBalanceLayer(/* args */);
    ~VkWhiteBalanceLayer();

   private:
    void parametTransform();

   protected:
    virtual void onUpdateParamet() override;
};

class VkChromaKeyLayer : public VkLayer, public ITLayer<ChromaKeyParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkChromaKeyLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   protected:
   public:
    VkChromaKeyLayer(/* args */);
    virtual ~VkChromaKeyLayer();
};

class VkColorInvertLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkColorInvertLayer)
   private:
    /* data */
   public:
    VkColorInvertLayer(/* args */);
    virtual ~VkColorInvertLayer();
};

class VkContrastLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkContrastLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkContrastLayer(/* args */);
    virtual ~VkContrastLayer();
};

class VkFalseColorLayer : public VkLayer, public ITLayer<FalseColorParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkFalseColorLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkFalseColorLayer(/* args */);
    virtual ~VkFalseColorLayer();
};

class VkHueLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkHueLayer)

   private:
    /* data */
   public:
    VkHueLayer(/* args */);
    virtual ~VkHueLayer();

   private:
    void transformParamet();

   protected:
    virtual void onUpdateParamet() override;
};

class VkLevelsLayer : public VkLayer, public ITLayer<LevelsParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkLevelsLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkLevelsLayer(/* args */);
    virtual ~VkLevelsLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce