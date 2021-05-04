#include "VkMotionBlurLayer.hpp"

#include "aoce_vulkan/layer/VkPipeGraph.hpp"

using namespace std::placeholders;

namespace aoce {
namespace vulkan {
namespace layer {

VkMotionBlurLayer::VkMotionBlurLayer(/* args */) {
    glslPath = "glsl/motionBlur.comp.spv";
    setUBOSize(sizeof(vec2));
}

VkMotionBlurLayer::~VkMotionBlurLayer() {}

void VkMotionBlurLayer::transformParamet() {
    assert(inFormats[0].width > 0 && inFormats[0].height > 0);
    float aspectRatio = (float)inFormats[0].height / inFormats[0].width;
    vec2 offset = {};
    offset.x = paramet.blurSize * cos(paramet.blurAngle * M_PI / 180.0) *
               aspectRatio / inFormats[0].width;
    offset.y = paramet.blurSize * sin(paramet.blurAngle * M_PI / 180.0) /
               inFormats[0].height;
    updateUBO(&offset);
}

bool VkMotionBlurLayer::getSampled(int32_t inIndex) { return inIndex == 0; }

void VkMotionBlurLayer::onInitLayer() {
    VkLayer::onInitLayer();
    transformParamet();
}

VkMotionDetectorLayer::VkMotionDetectorLayer(/* args */) {
    glslPath = "glsl/motionDetector.comp.spv";
    inCount = 2;

    lowLayer = std::make_unique<VkLowPassLayer>();
    avageLayer = std::make_unique<VkReduceLayer>(ReduceOperate::sum);
    outLayer = std::make_unique<VkOutputLayer>();

    paramet = 0.5f;
    lowLayer->updateParamet(paramet);
    outLayer->setImageProcessHandle(std::bind(
        &VkMotionDetectorLayer::onImageProcessHandle, this, _1, _2, _3));
}

void VkMotionDetectorLayer::onImageProcessHandle(uint8_t* data,
                                                 ImageFormat imageFormat,
                                                 int32_t outIndex) {
    if (onMotionEvent) {
        vec4 motion = {};
        memcpy(&motion, data, sizeof(vec4));
        float size = inFormats[0].width*inFormats[0].height;
        motion.z = motion.z / size;
        motion.w = motion.w / size;
        onMotionEvent(motion);
    }
}

VkMotionDetectorLayer::~VkMotionDetectorLayer() {}

void VkMotionDetectorLayer::onUpdateParamet() {
    lowLayer->updateParamet(paramet);
}

void VkMotionDetectorLayer::onInitGraph() {
    VkLayer::onInitGraph();
    pipeGraph->addNode(lowLayer->getLayer());
    pipeGraph->addNode(avageLayer.get())->addNode(outLayer->getLayer());
}

void VkMotionDetectorLayer::onInitNode() {
    lowLayer->addLine(this, 0, 1);
    this->addLine(avageLayer.get());
    setStartNode(this, 0);
    setStartNode(lowLayer.get());
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce