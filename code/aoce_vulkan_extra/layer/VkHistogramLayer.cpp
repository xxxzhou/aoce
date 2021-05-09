#include "VkHistogramLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkHistogramLayer::VkHistogramLayer(bool signalChannal) {
    glslPath = "glsl/histogramC1.comp.spv";
    if (!signalChannal) {
        this->channalCount = 4;
        outCount = 4;
        glslPath = "glsl/histogram.comp.spv";
    }
}

VkHistogramLayer::~VkHistogramLayer() {}

void VkHistogramLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    if (channalCount > 1) {
        inFormats[0].imageType = ImageType::rgba8;
    }
    for (int32_t i = 0; i < channalCount; i++) {
        outFormats[i].imageType = ImageType::r32;
    }
}

void VkHistogramLayer::onInitLayer() {
    VkLayer::onInitLayer();
    for (int32_t i = 0; i < channalCount; i++) {
        outFormats[i].width = 256;
        outFormats[i].height = 1;
    }
}

void VkHistogramLayer::onCommand() {
    clearColor({0, 0, 0, 0});
    VkLayer::onCommand();
}

VkHistogramC4Layer::VkHistogramC4Layer(/* args */) {
    glslPath = "glsl/histogramCombin.comp.spv";
    preLayer = std::make_unique<VkHistogramLayer>(false);
    inCount = 4;
}

VkHistogramC4Layer::~VkHistogramC4Layer() {}

void VkHistogramC4Layer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r32;
    inFormats[1].imageType = ImageType::r32;
    inFormats[2].imageType = ImageType::r32;
    inFormats[3].imageType = ImageType::r32;
    outFormats[0].imageType = ImageType::rgba32;

    pipeGraph->addNode(preLayer.get());
}

void VkHistogramC4Layer::onInitNode() {
    preLayer->addLine(this, 0, 0);
    preLayer->addLine(this, 1, 1);
    preLayer->addLine(this, 2, 2);
    preLayer->addLine(this, 3, 3);
    setStartNode(preLayer.get());
}

void VkHistogramC4Layer::onInitLayer() {
    sizeX = divUp(inFormats[0].width, 256);
    sizeY = 1;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce