#include "VkMorphLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkPreDilationLayer::VkPreDilationLayer(bool bSingle) {
    glslPath = "glsl/morph1_dilation.comp.spv";
    this->bSingle = bSingle;
    if (bSingle) {
        glslPath = "glsl/morph1_dilationC1.comp.spv";
    }
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
}

VkPreDilationLayer::~VkPreDilationLayer() {}

void VkPreDilationLayer::onInitGraph() {
    VkLayer::onInitGraph();
    if (bSingle) {
        inFormats[0].imageType = ImageType::r8;
        outFormats[0].imageType = ImageType::r8;
    }
}

VkDilationLayer::VkDilationLayer(bool bSingle) {
    glslPath = "glsl/morph2_dilation.comp.spv";
    this->bSingle = bSingle;
    if (bSingle) {
        glslPath = "glsl/morph2_dilationC1.comp.spv";
    }
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
    preLayer = std::make_unique<VkPreDilationLayer>(bSingle);
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
    if (bSingle) {
        inFormats[0].imageType = ImageType::r8;
        outFormats[0].imageType = ImageType::r8;
    }
    pipeGraph->addNode(preLayer->getLayer());
}

void VkDilationLayer::onInitNode() {
    preLayer->addLine(this, 0, 0);
    setStartNode(preLayer.get());
}

VkPreErosionLayer::VkPreErosionLayer(bool bSingle) {
    glslPath = "glsl/morph1_erosion.comp.spv";
    this->bSingle = bSingle;
    if (bSingle) {
        glslPath = "glsl/morph1_erosionC1.comp.spv";
    }
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
}

VkPreErosionLayer::~VkPreErosionLayer() {}

void VkPreErosionLayer::onInitGraph() {
    VkLayer::onInitGraph();
    if (bSingle) {
        inFormats[0].imageType = ImageType::r8;
        outFormats[0].imageType = ImageType::r8;
    }
}

VkErosionLayer::VkErosionLayer(bool bSingle) {
    glslPath = "glsl/morph2_erosion.comp.spv";
    this->bSingle = bSingle;
    if (bSingle) {
        glslPath = "glsl/morph2_erosionC1.comp.spv";
    }
    setUBOSize(sizeof(paramet), true);
    paramet = 3;
    updateUBO(&paramet);
    preLayer = std::make_unique<VkPreErosionLayer>(bSingle);
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
    if (bSingle) {
        inFormats[0].imageType = ImageType::r8;
        outFormats[0].imageType = ImageType::r8;
    }
    pipeGraph->addNode(preLayer->getLayer());
}

void VkErosionLayer::onInitNode() {
    preLayer->addLine(this, 0, 0);
    setStartNode(preLayer.get());
}

VkClosingLayer::VkClosingLayer(bool bSingle) {
    dilationLayer = std::make_unique<VkDilationLayer>(bSingle);
    erosionLayer = std::make_unique<VkErosionLayer>(bSingle);
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

VkOpeningLayer::VkOpeningLayer(bool bSingle) {
    dilationLayer = std::make_unique<VkDilationLayer>(bSingle);
    erosionLayer = std::make_unique<VkErosionLayer>(bSingle);
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