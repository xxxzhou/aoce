#include "VkVoronoiLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkVoronoiConsumerLayer::VkVoronoiConsumerLayer(/* args */) {
    glslPath = "glsl/voronoiConsumer.comp.spv";
    inCount = 2;
}

VkVoronoiConsumerLayer::~VkVoronoiConsumerLayer() {}

bool VkVoronoiConsumerLayer::getSampled(int32_t inIndex) {
    return inIndex == 0 || inIndex == 1;
}

void VkVoronoiConsumerLayer::onInitLayer() {
    float width = inFormats[1].width;
    float height = inFormats[1].height;

    width = log2(width);
    height = log2(height);

    logAssert(width == height, "Voronoi point texture must be square");
    logAssert(width == floor(width) && height == floor(height),
              "Voronoi point texture must be a power of 2");
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce