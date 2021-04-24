#include "VkLookupLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLookupLayer::VkLookupLayer(/* args */) {
    glslPath = "glsl/lookup.comp.spv";
    inCount = 2;
    outCount = 1;
    lookupLayer = std::make_unique<VkInputLayer>();
}

VkLookupLayer::~VkLookupLayer() {}

void VkLookupLayer::loadLookUp(uint8_t* data, int32_t size) {
    VideoType vtype = VideoType::other;
    if (size == 512 * 512 * 4) {
        vtype = VideoType::rgba8;
    } else if (size == 512 * 512 * 3) {
        vtype = VideoType::rgb8;
    } else {
        logAssert(vtype != VideoType::other, "look up table image error");
        assert(vtype != VideoType::other);
    }
    VideoFormat vformat = {};
    vformat.width = 512;
    vformat.height = 512;
    vformat.videoType = vtype;
    lookupLayer->setImage(vformat);
    lookupLayer->inputCpuData(data);
}

bool VkLookupLayer::getSampled(int inIndex) {
    if (inIndex == 1) {
        return true;
    }
    return false;
}

void VkLookupLayer::onInitGraph() {
    VkLayer::onInitGraph();
    pipeGraph->addNode(lookupLayer->getLayer());
}

void VkLookupLayer::onInitNode() {
    lookupLayer->getLayerNode()->addLine(getNode(), 0, 1);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce