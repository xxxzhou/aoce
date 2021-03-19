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
    inFormats[0].imageType = ImageType::rgbaf32;
    outFormats[0].imageType = ImageType::rgbaf32;
    outFormats[1].imageType = ImageType::rgbaf32;
    outFormats[2].imageType = ImageType::rgbaf32;
    VkLayer::onInitGraph();
}

VkGuidedLayer::VkGuidedLayer(/* args */) {
    setUBOSize(4);
    convertLayer = std::make_unique<VkConvertImageLayer>();
    resizeLayer = std::make_unique<VkResizeLayer>(ImageType::rgbaf32);
    resizeLayer->updateParamet({false, 1920 / 8, 1080 / 8});
    toMatLayer = std::make_unique<VkToMatLayer>();
    box1Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgbaf32);
    box2Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgbaf32);
    box3Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgbaf32);
    box4Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgbaf32);
    box5Layer = std::make_unique<VkBoxBlurSLayer>(ImageType::rgbaf32);
    // self
    glslPath = "glsl/guidedFilter2.comp.spv";
    resize1Layer = std::make_unique<VkResizeLayer>(ImageType::rgbaf32);
    resize1Layer->updateParamet({true, 1920, 1080});

    inCount = 4;
    outCount = 1;
}

void VkGuidedLayer::onUpdateParamet() {
    if (paramet.boxSize != oldParamet.boxSize) {
        box1Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box2Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box3Layer->updateParamet({paramet.boxSize, paramet.boxSize});
        box4Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    }
    if (paramet.eps != oldParamet.eps) {
        memcpy(constBufCpu.data(), &paramet.eps, conBufSize);
        bParametChange = true;
    }
}

void VkGuidedLayer::onInitGraph() {
    VkLayer::onInitGraph();
    // 输入输出
    inFormats[0].imageType = ImageType::rgbaf32;
    inFormats[1].imageType = ImageType::rgbaf32;
    inFormats[2].imageType = ImageType::rgbaf32;
    inFormats[3].imageType = ImageType::rgbaf32;
    outFormats[0].imageType = ImageType::rgbaf32;
    pipeGraph->addNode(convertLayer.get())
        ->addNode(resizeLayer->getLayer())
        ->addNode(toMatLayer.get());
    pipeGraph->addNode(box1Layer->getLayer());
    pipeGraph->addNode(box2Layer->getLayer());
    pipeGraph->addNode(box3Layer->getLayer());
    pipeGraph->addNode(box4Layer->getLayer());
    pipeGraph->addNode(box5Layer->getLayer());
    pipeGraph->addNode(resize1Layer->getLayer());
    // 更新下默认UBO信息
    memcpy(constBufCpu.data(), &paramet.eps, conBufSize);
}

void VkGuidedLayer::onInitNode() {
    resizeLayer->getNode()->addLine(box1Layer->getNode(), 0, 0);
    toMatLayer->getNode()->addLine(box2Layer->getNode(), 0, 0);
    toMatLayer->getNode()->addLine(box3Layer->getNode(), 1, 0);
    toMatLayer->getNode()->addLine(box4Layer->getNode(), 2, 0);
    box1Layer->getNode()->addLine(getNode(), 0, 0);
    box2Layer->getNode()->addLine(getNode(), 0, 1);
    box3Layer->getNode()->addLine(getNode(), 0, 2);
    box4Layer->getNode()->addLine(getNode(), 0, 3);
    getNode()->addLine(box5Layer->getNode());
    box5Layer->getNode()->addLine(resize1Layer->getNode());
    getNode()->setStartNode(convertLayer->getNode());
    getNode()->setEndNode(resize1Layer->getNode());
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
    box1Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    box2Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    box3Layer->updateParamet({paramet.boxSize, paramet.boxSize});
    box4Layer->updateParamet({paramet.boxSize, paramet.boxSize});
}

VkGuidedLayer::~VkGuidedLayer() {}

VkGuidedMattingLayer::VkGuidedMattingLayer(BaseLayer* baseLayer) {
    inCount = 2;
    outCount = 1;
    guidedLayer = std::make_unique<VkGuidedLayer>();
    mattingLayer = baseLayer;
    glslPath = "glsl/guidedMatting.comp.spv";
}

VkGuidedMattingLayer::~VkGuidedMattingLayer() {}

void VkGuidedMattingLayer::onUpdateParamet() {
    guidedLayer->updateParamet(paramet.guided);
}

void VkGuidedMattingLayer::onInitGraph() {
    VkLayer::onInitGraph();
    // 输入输出
    inFormats[0].imageType = ImageType::rgba8;
    inFormats[1].imageType = ImageType::rgbaf32;
    outFormats[0].imageType = ImageType::rgba8;
    pipeGraph->addNode(mattingLayer)->addNode(guidedLayer->getLayer());
}

void VkGuidedMattingLayer::onInitNode() {
    mattingLayer->getNode()->addLine(getNode(), 0, 0);
    guidedLayer->getNode()->getEndNode()->addLine(getNode(), 0, 1);
    getNode()->setStartNode(mattingLayer->getNode());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce