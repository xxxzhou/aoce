#include "VkGuidedLayer.hpp"

#include "aoce/Layer/PipeGraph.hpp"
#include "aoce/Layer/PipeNode.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkToMatLayer::VkToMatLayer() {
    glslPath = "glsl/guidedFilter1.comp.spv";
    inCount = 1;
    outCount = 3;
}

VkToMatLayer::~VkToMatLayer() {}

void VkToMatLayer::onInitGraph() {
    inFormats[0].imageType = ImageType::rgba32f;
    outFormats[0].imageType = ImageType::rgba32f;
    outFormats[1].imageType = ImageType::rgba32f;
    outFormats[2].imageType = ImageType::rgba32f;
    VkLayer::onInitGraph();
}

VkGuidedSolveLayer::VkGuidedSolveLayer() {
    setUBOSize(sizeof(float), true);
    glslPath = "glsl/guidedFilter2.comp.spv";
    inCount = 4;
    outCount = 1;
    paramet = 0.000001f;
    updateUBO(&paramet);
}

VkGuidedSolveLayer::~VkGuidedSolveLayer() {}

void VkGuidedSolveLayer::onInitGraph() {
    inFormats[0].imageType = ImageType::rgba32f;
    inFormats[1].imageType = ImageType::rgba32f;
    inFormats[2].imageType = ImageType::rgba32f;
    inFormats[3].imageType = ImageType::rgba32f;
    outFormats[0].imageType = ImageType::rgba32f;
    VkLayer::onInitGraph();
}

VkGuidedLayer::VkGuidedLayer(/* args */) {
    // self
    glslPath = "glsl/guidedMatting.comp.spv";
    inCount = 2;
    outCount = 1;
    //
    convertLayer = std::make_unique<VkConvertImageLayer>();
    resizeLayer = std::make_unique<VkResizeLayer>(ImageType::rgba32f);    
    toMatLayer = std::make_unique<VkToMatLayer>();
    box1Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgba32f);
    box2Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgba32f);
    box3Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgba32f);
    box4Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgba32f);
    guidedSlayerLayer = std::make_unique<VkGuidedSolveLayer>();
    box5Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgba32f);
    resize1Layer = std::make_unique<VkResizeLayer>(ImageType::rgba32f);    
    //box1Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    resizeLayer->updateParamet({false, 1920 / 8, 1080 / 8});
    box2Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    box3Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    box4Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    box5Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    guidedSlayerLayer->updateParamet(paramet.eps);
    resize1Layer->updateParamet({true, 1920, 1080});
}

void VkGuidedLayer::onUpdateParamet() {
    if (paramet.boxSize != oldParamet.boxSize) {
        box1Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box2Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box3Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box4Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box5Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    }
    if (paramet.eps != oldParamet.eps) {
        guidedSlayerLayer->updateParamet(paramet.eps);
    }
}

void VkGuidedLayer::onInitGraph() {
    VkLayer::onInitGraph();
    // 输入输出
    inFormats[0].imageType = ImageType::rgba32f;
    inFormats[1].imageType = ImageType::rgba32f;
    outFormats[0].imageType = ImageType::rgba8;
    pipeGraph->addNode(convertLayer.get())
        ->addNode(resizeLayer->getLayer())
        ->addNode(toMatLayer.get());
    pipeGraph->addNode(box1Layer->getLayer());
    pipeGraph->addNode(box2Layer->getLayer());
    pipeGraph->addNode(box3Layer->getLayer());
    pipeGraph->addNode(box4Layer->getLayer());
    pipeGraph->addNode(guidedSlayerLayer->getLayer());
    pipeGraph->addNode(box5Layer->getLayer());
    pipeGraph->addNode(resize1Layer->getLayer());
}

void VkGuidedLayer::onInitNode() {
    resizeLayer->getNode()->addLine(box1Layer->getNode(), 0, 0);
    toMatLayer->getNode()->addLine(box2Layer->getNode(), 0, 0);
    toMatLayer->getNode()->addLine(box3Layer->getNode(), 1, 0);
    toMatLayer->getNode()->addLine(box4Layer->getNode(), 2, 0);
    box1Layer->getNode()->addLine(guidedSlayerLayer->getNode(), 0, 0);
    box2Layer->getNode()->addLine(guidedSlayerLayer->getNode(), 0, 1);
    box3Layer->getNode()->addLine(guidedSlayerLayer->getNode(), 0, 2);
    box4Layer->getNode()->addLine(guidedSlayerLayer->getNode(), 0, 3);
    guidedSlayerLayer->getNode()->addLine(box5Layer->getNode());
    box5Layer->getNode()->addLine(resize1Layer->getNode());
    convertLayer->getNode()->addLine(getNode(), 0, 0);
    resize1Layer->getNode()->addLine(getNode(), 0, 1);
    getNode()->setStartNode(convertLayer->getNode());
}

void VkGuidedLayer::onInitLayer() {
    VkLayer::onInitLayer();
    ImageFormat format = {};
    logAssert(resizeLayer->getInFormat(format),
              "guider layer get error image format");
    int32_t width = format.width;
    int32_t height = format.height;
    int32_t scaleWidth = divUp(width, zoom);
    int32_t scaleHeight = divUp(height, zoom);
    resizeLayer->updateParamet({false, scaleWidth, scaleHeight});
    resize1Layer->updateParamet({true, width, height});    
}

VkGuidedLayer::~VkGuidedLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce