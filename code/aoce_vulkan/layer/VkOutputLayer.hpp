#pragma once
#include <Layer/OutputLayer.hpp>
#if __ANDROID__
#include "../android/HardwareImage.hpp"
#endif
#include "VkLayer.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

class VkOutputLayer : public OutputLayer, public VkLayer {
    AOCE_LAYER_QUERYINTERFACE(VkOutputLayer)
   private:
    // CPU输出使用
    std::unique_ptr<VulkanBuffer> outBuffer = nullptr;
    std::vector<uint8_t> cpuData;
    VkOutGpuTex outTex = {};
    // std::unique_ptr<VulkanTexture> outTex = nullptr;
#if __ANDROID__
    std::unique_ptr<HardwareImage> hardwareImage = nullptr;
#endif
    VkEvent outEvent = VK_NULL_HANDLE;

   public:
    VkOutputLayer(/* args */);
    ~VkOutputLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitVkBuffer() override;
    virtual void onPreCmd() override;
    virtual bool onFrame() override;

   public:
    virtual void outGpuTex(const VkOutGpuTex& outTex,
                           int32_t outIndex = 0) override;

#if __ANDROID__
    virtual void outGLGpuTex(const VkOutGpuTex& outTex,uint32_t texType = 0,
                             int32_t outIndex = 0) override;
#endif
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce