#include "VkMorphLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPreDilationLayer::VkPreDilationLayer() {
    glslPath = "glsl/morph1_dilation.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
}

VkPreDilationLayer::~VkPreDilationLayer() {}

void VkPreDilationLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
}

VkDilationLayer::VkDilationLayer() {
    glslPath = "glsl/morph2_dilation.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
    preLayer = std::make_unique<VkPreDilationLayer>();
}

VkDilationLayer::~VkDilationLayer() {}

void VkDilationLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    preLayer->updateParamet(paramet);
    updateUBO(&paramet);
    bParametChange = true;
}

void VkDilationLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
    pipeGraph->addNode(preLayer->getLayer());
}

void VkDilationLayer::onInitNode() {
    preLayer->addLine(this, 0, 0);
    setStartNode(preLayer.get());
}

VkPreErosionLayer::VkPreErosionLayer() {
    glslPath = "glsl/morph1_erosion.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
}

VkPreErosionLayer::~VkPreErosionLayer() {}

void VkPreErosionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
}

VkErosionLayer::VkErosionLayer() {
    glslPath = "glsl/morph2_erosion.comp.spv";
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
    preLayer = std::make_unique<VkPreErosionLayer>();
}

VkErosionLayer::~VkErosionLayer() {}

void VkErosionLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    preLayer->updateParamet(paramet);
    updateUBO(&paramet);
    bParametChange = true;
}

void VkErosionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r8;
    outFormats[0].imageType = ImageType::r8;
    pipeGraph->addNode(preLayer->getLayer());
}

void VkErosionLayer::onInitNode() {
    preLayer->addLine(this, 0, 0);
    setStartNode(preLayer.get());
}

VkClosingLayer::VkClosingLayer() {
    dilationLayer = std::make_unique<VkDilationLayer>();
    erosionLayer = std::make_unique<VkErosionLayer>();
}

VkClosingLayer::~VkClosingLayer() {}

void VkClosingLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    dilationLayer->updateParamet(paramet);
    erosionLayer->updateParamet(paramet);
}

void VkClosingLayer::onInitNode() {
    pipeGraph->addNode(dilationLayer->getLayer())
        ->addNode(erosionLayer->getLayer());
    setStartNode(dilationLayer.get());
    setEndNode(erosionLayer.get());
}

VkOpeningLayer::VkOpeningLayer() {
    dilationLayer = std::make_unique<VkDilationLayer>();
    erosionLayer = std::make_unique<VkErosionLayer>();
}

VkOpeningLayer::~VkOpeningLayer() {}

void VkOpeningLayer::onUpdateParamet() {
    if (paramet == oldParamet) {
        return;
    }
    dilationLayer->updateParamet(paramet);
    erosionLayer->updateParamet(paramet);
}

void VkOpeningLayer::onInitNode() {
    pipeGraph->addNode(erosionLayer->getLayer())
        ->addNode(dilationLayer->getLayer());
    setStartNode(erosionLayer.get());
    setEndNode(dilationLayer.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce