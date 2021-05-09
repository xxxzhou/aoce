#pragma once
#include <Layer/OutputLayer.hpp>
#if WIN32
#include "../win32/VkWinImage.hpp"
#elif __ANDROID__
#include "../android/HardwareImage.hpp"
#endif
#include "VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class AOCE_VULKAN_EXPORT VkOutputLayer : public OutputLayer, public VkLayer {
    AOCE_LAYER_QUERYINTERFACE(VkOutputLayer)
   private:
    // CPU输出使用
    std::unique_ptr<VulkanBuffer> outBuffer = nullptr;
    std::vector<uint8_t> cpuData;
    VkOutGpuTex outTex = {};
    std::unique_ptr<VulkanTexture> swapTex = nullptr;
#if WIN32
    std::unique_ptr<VkWinImage> winImage;
    bool bWinInterop = false;
#elif __ANDROID__
    std::unique_ptr<HardwareImage> hardwareImage = nullptr;
#endif

   public:
    VkOutputLayer(/* args */);
    ~VkOutputLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onUpdateParamet() override;
    virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override{};
    virtual void onCommand() override;
    virtual bool onFrame() override;

   public:
    virtual void outVkGpuTex(const VkOutGpuTex& outTex,
                             int32_t outIndex = 0) override;

// android api < 26不能使用GPU数据转换,只能用CPU数据传输
#if __ANDROID__
    virtual void outGLGpuTex(const VkOutGpuTex& outTex, uint32_t texType = 0,
                             int32_t outIndex = 0) override;
#endif
#if WIN32
    virtual void outDx11GpuTex(void* device, void* tex) override;
#endif
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce