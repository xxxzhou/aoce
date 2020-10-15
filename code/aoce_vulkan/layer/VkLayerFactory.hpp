#pragma once

#include <Layer/LayerFactory.hpp>

namespace aoce {
namespace vulkan {
namespace layer {

class VkLayerFactory : public LayerFactory {
   private:
    /* data */
   public:
    VkLayerFactory(/* args */);
    virtual ~VkLayerFactory() override;

   public:
    virtual InputLayer* crateInput() override;
    virtual OutputLayer* createOutput() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce