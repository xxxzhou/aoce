#pragma once

#include <Layer/PipeGraph.hpp>
#include <memory>

#include "../vulkan/VulkanContext.hpp"
#include "VkLayer.hpp"

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
    VkFence computerFence;

   public:
    VkPipeGraph(/* args */);
    ~VkPipeGraph();

   public:
    inline VulkanContext* getContext() { return context.get(); };

    VulkanTexturePtr getOutTex(int32_t node, int32_t outIndex);

   public:
    // 所有layer调用initbuffer后
    virtual bool onInitBuffers();
    virtual bool onRun();
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
