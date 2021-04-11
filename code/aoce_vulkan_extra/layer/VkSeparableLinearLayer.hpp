#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 边框默认使用REPLICATE模式
class VkSeparableLayer : public VkLayer {
   protected:
    /* data */
    ImageType imageType = ImageType::rgba8;
    std::unique_ptr<VulkanBuffer> kernelBuffer;

   public:
    VkSeparableLayer(ImageType imageType = ImageType::rgba8);
    ~VkSeparableLayer();

    void updateBuffer(std::vector<float> data);

   protected:
    virtual void onInitGraph() override;
    virtual void onInitLayer() override;
    virtual void onInitPipe() override;
};

class VkSeparableLinearLayer : public VkSeparableLayer {
   protected:
    std::unique_ptr<VkSeparableLayer> rowLayer;

   public:
    VkSeparableLinearLayer(ImageType imageType = ImageType::rgba8);
    ~VkSeparableLinearLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
};

class VkBoxBlurSLayer : public VkSeparableLinearLayer,
                        public ITLayer<KernelSizeParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkBoxBlurSLayer)

   public:
    VkBoxBlurSLayer(ImageType imageType = ImageType::rgba8);
    virtual ~VkBoxBlurSLayer();

   public:
    void getKernel(int size, std::vector<float>& kernels);

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
};

class VkGaussianBlurSLayer : public VkSeparableLinearLayer,
                             public ITLayer<GaussianBlurParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkGaussianBlurSLayer)

   public:
    VkGaussianBlurSLayer(ImageType imageType = ImageType::rgba8);
    virtual ~VkGaussianBlurSLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
};


}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
