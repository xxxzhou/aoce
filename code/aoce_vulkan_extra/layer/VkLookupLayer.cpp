#include "VkLookupLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"

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
    lookupLayer->getLayer()->addLine(this, 0, 1);
}

VkSoftEleganceLayer::VkSoftEleganceLayer(/* args */) {
    lookupLayer1 = std::make_unique<VkLookupLayer>();
    lookupLayer2 = std::make_unique<VkLookupLayer>();
    blurLayer = std::make_unique<VkGaussianBlurSLayer>();
    alphaBlendLayer = std::make_unique<VkAlphaBlendLayer>();

    onUpdateParamet();
}

VkSoftEleganceLayer::~VkSoftEleganceLayer() {}

void VkSoftEleganceLayer::loadLookUp1(uint8_t* data, int32_t size) {
    lookupLayer1->loadLookUp(data, size);
}

void VkSoftEleganceLayer::loadLookUp2(uint8_t* data, int32_t size) {
    lookupLayer2->loadLookUp(data, size);
}

void VkSoftEleganceLayer::onUpdateParamet() {
    blurLayer->updateParamet(paramet.blur);
    alphaBlendLayer->updateParamet(paramet.mix);
}

void VkSoftEleganceLayer::onInitNode() {
    pipeGraph->addNode(lookupLayer1->getLayer())
        ->addNode(alphaBlendLayer->getLayer())
        ->addNode(lookupLayer2->getLayer());
    pipeGraph->addNode(blurLayer->getLayer());
    lookupLayer1->addLine(blurLayer.get());
    blurLayer->addLine(alphaBlendLayer.get(), 0, 1);

    setStartNode(lookupLayer1.get());
    setEndNode(lookupLayer2.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce