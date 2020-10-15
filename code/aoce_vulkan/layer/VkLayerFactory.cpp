#include "VkLayerFactory.hpp"

#include "VkInputLayer.hpp"
#include "VkOutputLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLayerFactory::VkLayerFactory(/* args */) {}

VkLayerFactory::~VkLayerFactory() {}

InputLayer* VkLayerFactory::crateInput() {
    InputLayer* inLayer = new VkInputLayer();
    return inLayer;
}

OutputLayer* VkLayerFactory::createOutput() { return new VkOutputLayer(); }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce