#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 边框默认使用REPLICATE模式
class VkLinearFilterLayer : public VkLayer, public ITLayer<BoxBlueParamet> {
    // AOCE_LAYER_QUERYINTERFACE(VkLinearFilterLayer)
   protected:
    std::unique_ptr<VulkanBuffer> kernelBuffer;
    bool bOneChannel = false;

   public:
    VkLinearFilterLayer(bool bOneChannel = false);
    virtual ~VkLinearFilterLayer();

   protected:
    virtual void onUpdateParamet() override;

    virtual void onInitGraph() override;
    // virtual void onInitLayer() override;
    // virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override;
};

class VkBoxBlurLayer : public VkLinearFilterLayer {
    AOCE_LAYER_QUERYINTERFACE(VkBoxBlurLayer)

   public:
    VkBoxBlurLayer(bool bOneChannel = false);
    virtual ~VkBoxBlurLayer();

   protected:
    virtual void onInitVkBuffer() override;
};

class VkGaussianBlurLayer : public VkLinearFilterLayer {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurLayer)

   public:
    VkGaussianBlurLayer(bool bOneChannel = false);
    virtual ~VkGaussianBlurLayer();

   protected:
    virtual void onInitVkBuffer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
