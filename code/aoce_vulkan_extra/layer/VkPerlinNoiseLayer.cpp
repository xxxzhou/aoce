#include "VkPerlinNoiseLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPerlinNoiseLayer::VkPerlinNoiseLayer(/* args */) {
    bInput = true;
    glslPath = "glsl/perlinNoise.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
    inCount = 0;
}

VkPerlinNoiseLayer::~VkPerlinNoiseLayer() {}

void VkPerlinNoiseLayer::setImageSize(int32_t width, int32_t height) {
    this->width = width;
    this->height = height;
}

void VkPerlinNoiseLayer::onInitGraph() {
    VkLayer::onInitGraph();
    if (outFormats[0].width != width || outFormats[0].height != height) {
        resetGraph();
    }
    outFormats[0].width = width;
    outFormats[0].height = height;
    outFormats[0].imageType = ImageType::rgba8;
}

void VkPerlinNoiseLayer::onInitLayer() {
    sizeX = divUp(outFormats[0].width, groupX);
    sizeY = divUp(outFormats[0].height, groupY);
}



}  // namespace layer
}  // namespace vulkan
}  // namespace aoce