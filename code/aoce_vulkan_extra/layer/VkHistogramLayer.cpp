#include "VkHistogramLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"

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

VkHistogramLutLayer::VkHistogramLutLayer() {
    glslPath = "glsl/histogramLut.comp.spv";
    setUBOSize(sizeof(paramet), true);
}

VkHistogramLutLayer::~VkHistogramLutLayer() {}

void VkHistogramLutLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = ImageType::r32;
    outFormats[0].imageType = ImageType::r32f;
}

void VkHistogramLutLayer::onInitLayer() {
    sizeX = divUp(inFormats[0].width, 256);
    sizeY = 1;
}

VkEqualizeHistLayer::VkEqualizeHistLayer(bool signalChannal) {
    bSignal = signalChannal;
    glslPath = "glsl/histogramLutResultC1.comp.spv";
    if (!bSignal) {
        glslPath = "glsl/histogramLutResult.comp.spv";
        lumLayer = std::make_unique<VkLuminanceLayer>();
    }
    histLayer = std::make_unique<VkHistogramLayer>(true);
    lutLayer = std::make_unique<VkHistogramLutLayer>();
    inCount = 2;
}

VkEqualizeHistLayer::~VkEqualizeHistLayer() {}

void VkEqualizeHistLayer::onInitGraph() {
    VkLayer::onInitGraph();
    inFormats[0].imageType = bSignal ? ImageType::r8 : ImageType::rgba8;
    inFormats[1].imageType = ImageType::r32f;
    outFormats[0].imageType = bSignal ? ImageType::r8 : ImageType::rgba8;
    if (bSignal) {
        pipeGraph->addNode(histLayer.get())->addNode(lutLayer->getLayer());
    } else {
        pipeGraph->addNode(lumLayer.get())
            ->addNode(histLayer.get())
            ->addNode(lutLayer->getLayer());
    }
}

void VkEqualizeHistLayer::onInitNode() {
    lutLayer->addLine(this, 0, 1);
    setStartNode(this);
    if (bSignal) {
        setStartNode(histLayer.get());
    } else {
        setStartNode(lumLayer.get());
    }
}

void VkEqualizeHistLayer::onInitLayer() {
    VkLayer::onInitLayer();
    ImageFormat iformt = {};
    pipeGraph->getLayerInFormat(histLayer->getGraphIndex(), 0, iformt);
    lutLayer->updateParamet(iformt.width * iformt.height);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce