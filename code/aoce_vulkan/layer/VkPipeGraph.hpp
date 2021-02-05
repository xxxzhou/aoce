#pragma once

#include <Layer/PipeGraph.hpp>
#include <memory>

#include "../vulkan/VulkanContext.hpp"
#include "VkLayer.hpp"

#if WIN32
#include "aoce_win/DX11/Dx11Helper.hpp"
#include "../win32/VkWinImage.hpp"
#endif

namespace aoce {
namespace vulkan {
namespace layer {

class AOCE_VULKAN_EXPORT VkPipeGraph : public PipeGraph {
   private:
    // 对应一个vkCommandBuffer
    std::unique_ptr<VulkanContext> context;
    // 输入层
    // std::vector<VkLayer*> vkInputLayers;
    // 输出层
    std::vector<VkLayer*> vkOutputLayers;
    // 余下层
    std::vector<VkLayer*> vkLayers;
    // GPU是否执行完成
    VkFence computerFence;

    bool delayGpu = false;    
    // 确定是否在重置生成资源与commandbuffer中
    VkEvent outEvent = VK_NULL_HANDLE;
    VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
#if WIN32    
    CComPtr<ID3D11Device> device = nullptr;
    CComPtr<ID3D11DeviceContext> ctx = nullptr;
    std::vector<VkDeviceMemory> outMemorys;
    std::vector<VkWinImage*> winImages;
    bool bDX11Update = false;
#endif
   public:
    VkPipeGraph(/* args */);
    ~VkPipeGraph();

   public:
    inline VulkanContext* getContext() { return context.get(); };
    VulkanTexturePtr getOutTex(int32_t node, int32_t outIndex);

    bool resourceReady();

#if WIN32
    ID3D11Device* getD3D11Device();
    void addOutMemory(VkWinImage* winImage);
#endif
    bool executeOut();


   protected:
    // 所有layer调用initbuffer后
    virtual void onReset() override;
    virtual bool onInitBuffers() override;
    virtual bool onRun() override;
};

class VkPipeGraphFactory : public PipeGraphFactory {
   public:
    VkPipeGraphFactory(){};
    virtual ~VkPipeGraphFactory(){};

   public:
    inline virtual PipeGraph* createGraph() override {
        return new VkPipeGraph();
    };
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce
