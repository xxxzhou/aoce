#include "VkLayerFactory.hpp"

#include "layer/VkInputLayer.hpp"
#include "layer/VkOutputLayer.hpp"
#include "layer/VkYUV2RGBALayer.hpp"
#include "layer/VkRGBA2YUVLayer.hpp"
#include "layer/VkOperateLayer.hpp"
#include "layer/VkTransposeLayer.hpp"
#include "layer/VkResizeLayer.hpp"
#include "layer/VkBlendLayer.hpp"

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

RGBA2YUVLayer* VkLayerFactory::createRGBA2YUV() {
    return new VkRGBA2YUVLayer();
}

TexOperateLayer* VkLayerFactory::createTexOperate() {
    return new VkOperateLayer();
}

TransposeLayer* VkLayerFactory::createTranspose() {
    return new VkTransposeLayer();
}

ReSizeLayer* VkLayerFactory::createSize() {
    return new VkResizeLayer();
}

BlendLayer* VkLayerFactory::createBlend() {
    return new VkBlendLayer();
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce