#include "VkLayerFactory.hpp"

#include "layer/VkBlendLayer.hpp"
#include "layer/VkInputLayer.hpp"
#include "layer/VkOperateLayer.hpp"
#include "layer/VkOutputLayer.hpp"
#include "layer/VkRGBA2YUVLayer.hpp"
#include "layer/VkResizeLayer.hpp"
#include "layer/VkTransposeLayer.hpp"
#include "layer/VkYUV2RGBALayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLayerFactory::VkLayerFactory(/* args */) {}

VkLayerFactory::~VkLayerFactory() {}

IInputLayer* VkLayerFactory::createInput() {
    InputLayer* inLayer = new VkInputLayer();
    return inLayer;
}

IOutputLayer* VkLayerFactory::createOutput() { return new VkOutputLayer(); }

IYUVLayer* VkLayerFactory::createYUV2RGBA() {
    return new VkYUV2RGBALayer();
}

IYUVLayer* VkLayerFactory::createRGBA2YUV() {
    return new VkRGBA2YUVLayer();
}

IMapChannelLayer* VkLayerFactory::createMapChannel() {
    return new VkMapChannelLayer();
}

IFlipLayer* VkLayerFactory::createFlip() { return new VkFlipLayer(); }

ITransposeLayer* VkLayerFactory::createTranspose() {
    return new VkTransposeLayer();
}

IReSizeLayer* VkLayerFactory::createSize() { return new VkResizeLayer(); }

IBlendLayer* VkLayerFactory::createBlend() { return new VkBlendLayer(); }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce