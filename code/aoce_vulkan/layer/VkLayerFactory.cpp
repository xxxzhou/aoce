#include "VkLayerFactory.hpp"

#include "VkInputLayer.hpp"

namespace aoce {
namespace vk {
namespace layer {

VkLayerFactory::VkLayerFactory(/* args */) {}

VkLayerFactory::~VkLayerFactory() {}

InputLayer* VkLayerFactory::crateInput() { return new VkInputLayer(); }

}  // namespace layer
}  // namespace vk
}  // namespace aoce