#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 边框默认使用REPLICATE模式
class VkLinearFilterLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkLinearFilterLayer)
    // AOCE_LAYER_QUERYINTERFACE(VkLinearFilterLayer)
   protected:
    std::unique_ptr<VulkanBuffer> kernelBuffer;
    // bool bOneChannel = false;
    ImageType imageType = ImageType::rgba8;

   public:
    VkLinearFilterLayer(ImageType imageType = ImageType::rgba8);
    virtual ~VkLinearFilterLayer();

   protected:
    virtual void onInitGraph() override;
    // virtual void onInitLayer() override;
    // virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override;
};

class VkBoxBlurLayer : public VkLinearFilterLayer,
                       public ITLayer<KernelSizeParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkBoxBlurLayer)

   public:
    VkBoxBlurLayer(ImageType imageType = ImageType::rgba8);
    virtual ~VkBoxBlurLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitVkBuffer() override;
};

class VkGaussianBlurLayer : public VkLinearFilterLayer,
                            public ITLayer<GaussianBlurParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurLayer)

   public:
    VkGaussianBlurLayer(ImageType imageType = ImageType::rgba8);
    virtual ~VkGaussianBlurLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitVkBuffer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
