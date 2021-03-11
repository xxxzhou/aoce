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
    bool bOneChannel = false;
    std::unique_ptr<VulkanBuffer> kernelBuffer;

   public:
    VkSeparableLayer(bool bOneChannel = false);
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
    VkSeparableLinearLayer(bool bOneChannel = false);
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
    VkBoxBlurSLayer(bool bOneChannel = false);
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
    VkGaussianBlurSLayer(bool bOneChannel = false);
    virtual ~VkGaussianBlurSLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
