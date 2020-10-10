#pragma once

#include <Layer/BaseLayer.hpp>

namespace aoce {
namespace vk {
namespace layer {

class VkLayer : public BaseLayer {
   private:
    /* data */
   protected:
    GpuType gpu = GpuType::vulkan;
    class VkPipeGraph* vkPipeGraph = nullptr;
    
   public:
    VkLayer(/* args */);
    ~VkLayer() override;

   public:
    virtual void onInit() override;
    virtual void onInitLayer() override;
    virtual void onRun() override;
};

}  // namespace layer
}  // namespace vk
}  // namespace aoce