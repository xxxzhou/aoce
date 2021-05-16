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

IInputLayer* VkLayerFactory::crateInput() {
    InputLayer* inLayer = new VkInputLayer();
    return inLayer;
}

IOutputLayer* VkLayerFactory::createOutput() { return new VkOutputLayer(); }

IYUV2RGBALayer* VkLayerFactory::createYUV2RGBA() {
    return new VkYUV2RGBALayer();
}

IRGBA2YUVLayer* VkLayerFactory::createRGBA2YUV() {
    return new VkRGBA2YUVLayer();
}

ITexOperateLayer* VkLayerFactory::createTexOperate() {
    return new VkOperateLayer();
}

ITransposeLayer* VkLayerFactory::createTranspose() {
    return new VkTransposeLayer();
}

IReSizeLayer* VkLayerFactory::createSize() { return new VkResizeLayer(); }

IBlendLayer* VkLayerFactory::createBlend() { return new VkBlendLayer(); }

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce