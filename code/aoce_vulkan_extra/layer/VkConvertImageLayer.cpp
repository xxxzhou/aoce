#include "VkConvertImageLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkConvertImageLayer::VkConvertImageLayer(ConvertType convert) {
    this->convert = convert;
    glslPath = "glsl/convertImage.comp.spv";
    if (convert == ConvertType::rgba32f2rgba8) {
        glslPath = "glsl/convertImageF4.comp.spv";
    }
}

VkConvertImageLayer::~VkConvertImageLayer() {}

void VkConvertImageLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::rgba32f;
    if (convert == ConvertType::rgba32f2rgba8) {
        inFormats[0].imageType = ImageType::rgba32f;
        outFormats[0].imageType = ImageType::rgba8;
    }
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce