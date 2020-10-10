#pragma once

#include <Layer/LayerFactory.hpp>

namespace aoce {
namespace vk {
namespace layer {

class VkLayerFactory : public LayerFactory {
   private:
    /* data */
   public:
    VkLayerFactory(/* args */);
    ~VkLayerFactory() override;

   public:
    virtual InputLayer* crateInput() override;
    virtual OutputLayer* createOutput() override { return nullptr; }
};

}  // namespace layer
}  // namespace vk
}  // namespace aoce