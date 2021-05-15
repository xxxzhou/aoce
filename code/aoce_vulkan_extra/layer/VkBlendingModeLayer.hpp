#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {
    
class VkAddBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkAddBlendLayer(/* args */);
    virtual ~VkAddBlendLayer();
};

class VkAlphaBlendLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkAlphaBlendLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkAlphaBlendLayer(/* args */);
    virtual ~VkAlphaBlendLayer();
};

class VkHardLightBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkHardLightBlendLayer(/* args */);
    ~VkHardLightBlendLayer();
};

class VkColorBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkColorBlendLayer)
   private:
    /* data */
   public:
    VkColorBlendLayer(/* args */);
    virtual ~VkColorBlendLayer();
};

class VkColorBurnBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkColorBurnBlendLayer)
   private:
    /* data */
   public:
    VkColorBurnBlendLayer(/* args */);
    virtual ~VkColorBurnBlendLayer();
};

class VkColorDodgeBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkColorDodgeBlendLayer)
   private:
    /* data */
   public:
    VkColorDodgeBlendLayer(/* args */);
    virtual ~VkColorDodgeBlendLayer();
};





class VkExclusionBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkExclusionBlendLayer)
   private:
    /* data */
   public:
    VkExclusionBlendLayer(/* args */);
    virtual ~VkExclusionBlendLayer();
};



class VkHueBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkHueBlendLayer)
   private:
    /* data */
   public:
    VkHueBlendLayer(/* args */);
    virtual ~VkHueBlendLayer();
};

class VkLightenBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkLightenBlendLayer)
   private:
    /* data */
   public:
    VkLightenBlendLayer(/* args */);
    virtual ~VkLightenBlendLayer();
};

class VkLinearBurnBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkLinearBurnBlendLayer)
   private:
    /* data */
   public:
    VkLinearBurnBlendLayer(/* args */);
    virtual ~VkLinearBurnBlendLayer();
};

class VkLuminosityBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkLuminosityBlendLayer)
   private:
    /* data */
   public:
    VkLuminosityBlendLayer(/* args */);
    virtual ~VkLuminosityBlendLayer();
};

class VkMaskLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkMaskLayer)
   private:
    /* data */
   public:
    VkMaskLayer(/* args */);
    virtual ~VkMaskLayer();
};

class VkMultiplyBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkMultiplyBlendLayer)
   private:
    /* data */
   public:
    VkMultiplyBlendLayer(/* args */);
    virtual ~VkMultiplyBlendLayer();
};

class VkNormalBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkNormalBlendLayer)
   private:
    /* data */
   public:
    VkNormalBlendLayer(/* args */);
    ~VkNormalBlendLayer();
};

class VkOverlayBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkOverlayBlendLayer)
   private:
    /* data */
   public:
    VkOverlayBlendLayer(/* args */);
    ~VkOverlayBlendLayer();
};

class VkSaturationBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkSaturationBlendLayer)
   private:
    /* data */
   public:
    VkSaturationBlendLayer(/* args */);
    ~VkSaturationBlendLayer();
};

class VkScreenBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkSaturationBlendLayer)
   private:
    /* data */
   public:
    VkScreenBlendLayer(/* args */);
    ~VkScreenBlendLayer();
};

class VkSoftLightBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkSoftLightBlendLayer)
   private:
    /* data */
   public:
    VkSoftLightBlendLayer(/* args */);
    ~VkSoftLightBlendLayer();
};

class VkSourceOverBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkSourceOverBlendLayer)
   private:
    /* data */
   public:
    VkSourceOverBlendLayer(/* args */);
    ~VkSourceOverBlendLayer();
};

class VkSubtractBlendLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkSoftLightBlendLayer)
   private:
    /* data */
   public:
    VkSubtractBlendLayer(/* args */);
    ~VkSubtractBlendLayer();
};

class VkDarkenBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkDarkenBlendLayer(/* args */);
    virtual ~VkDarkenBlendLayer();
};

class VkDifferenceBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkDifferenceBlendLayer(/* args */);
    virtual ~VkDifferenceBlendLayer();
};

class VkDissolveBlendLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkDissolveBlendLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkDissolveBlendLayer(/* args */);
    virtual ~VkDissolveBlendLayer();
};

class VkDivideBlendLayer : public VkLayer {
   private:
    /* data */
   public:
    VkDivideBlendLayer(/* args */);
    virtual ~VkDivideBlendLayer();
};


}
}
}