#include "VkLayerFactory.hpp"

#include "layer/VkInputLayer.hpp"
#include "layer/VkOutputLayer.hpp"
#include "layer/VkYUV2RGBALayer.hpp"

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

YUV2RGBALayer* VkLayerFactory::createYUV2RGBA() {
    return new VkYUV2RGBALayer();
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce