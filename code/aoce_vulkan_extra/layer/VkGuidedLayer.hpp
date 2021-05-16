#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkConvertImageLayer.hpp"
#include "VkLinearFilterLayer.hpp"
#include "VkSeparableLinearLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"
#include "aoce_vulkan/layer/VkResizeLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 导向滤波求值 Guided filter
// 论文地址http://kaiminghe.com/publications/pami12guidedfilter.pdf
// https://www.cnblogs.com/zhouxin/p/10203954.html
class VkToMatLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkToMatLayer)
   public:
    VkToMatLayer();
    virtual ~VkToMatLayer();

   protected:
    virtual void onInitGraph() override;
};

class VkGuidedSolveLayer : public VkLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkGuidedSolveLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkGuidedSolveLayer();
    virtual ~VkGuidedSolveLayer();

   protected:
    virtual void onInitGraph() override;
};

class VkGuidedLayer : public VkLayer, public ITLayer<GuidedParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGuidedLayer)
   private:
    /* data */
    std::unique_ptr<VkConvertImageLayer> convertLayer = nullptr;
    std::unique_ptr<VkResizeLayer> resizeLayer = nullptr;
    std::unique_ptr<VkToMatLayer> toMatLayer = nullptr;
    std::unique_ptr<VkBoxBlurSLayer> box1Layer = nullptr;
    std::unique_ptr<VkBoxBlurSLayer> box2Layer = nullptr;
    std::unique_ptr<VkBoxBlurSLayer> box3Layer = nullptr;
    std::unique_ptr<VkBoxBlurSLayer> box4Layer = nullptr;
    std::unique_ptr<VkGuidedSolveLayer> guidedSlayerLayer = nullptr;
    std::unique_ptr<VkBoxBlurSLayer> box5Layer = nullptr;
    std::unique_ptr<VkResizeLayer> resize1Layer = nullptr;

    const int32_t zoom = 8;

   public:
    VkGuidedLayer(/* args */);
    virtual ~VkGuidedLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce