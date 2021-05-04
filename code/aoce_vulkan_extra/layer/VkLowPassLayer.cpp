#include "VkLowPassLayer.hpp"

#include "aoce_vulkan/layer/VkPipeGraph.hpp"
#include "aoce_vulkan/vulkan/VulkanManager.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkSaveFrameLayer::VkSaveFrameLayer(/* args */) {
    bUserPipe = true;
    // 告诉外面,不需要自动连接别层输入
    bInput = true;
    glslPath = "glsl/copyImage.comp.spv";
}

VkSaveFrameLayer::~VkSaveFrameLayer() {}

void VkSaveFrameLayer::setImageFormat(const ImageFormat& imageFormat,
                                      int32_t nodeIndex, int32_t outNodeIndex) {
    inFormats[0] = imageFormat;
    outFormats[0] = imageFormat;
    inLayers[0].nodeIndex = nodeIndex;
    inLayers[0].siteIndex = outNodeIndex;
}

void VkSaveFrameLayer::onPreCmd() {
    inTexs.clear();
    inTexs.push_back(
        vkPipeGraph->getOutTex(inLayers[0].nodeIndex, inLayers[0].siteIndex));
    if (bUserPipe) {
        VkLayer::onInitLayer();
        VkLayer::onInitPipe();
        VkLayer::onPreCmd();
    } else {
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT);
        VulkanManager::copyImage(cmd, inTexs[0].get(), outTexs[0]->image);
    }
}

VkLowPassLayer::VkLowPassLayer(/* args */) {
    saveLayer = std::make_unique<VkSaveFrameLayer>();
}

VkLowPassLayer::~VkLowPassLayer() {}

void VkLowPassLayer::onInitGraph() {
    VkDissolveBlendLayer::onInitGraph();
    pipeGraph->addNode(saveLayer.get());
}

void VkLowPassLayer::onInitNode() {
    saveLayer->addLine(this, 0, 1);
}

void VkLowPassLayer::onInitLayer() {
    VkLayer::onInitLayer();
    saveLayer->setImageFormat(inFormats[0], getGraphIndex(), 0);
}

VkHighPassLayer::VkHighPassLayer(/* args */) {
    lowLayer = std::make_unique<VkLowPassLayer>();
    paramet = 0.5f;
    lowLayer->updateParamet(paramet);
}

VkHighPassLayer::~VkHighPassLayer() {}

void VkHighPassLayer::onUpdateParamet() { lowLayer->updateParamet(paramet); }

void VkHighPassLayer::onInitGraph() {
    VkDifferenceBlendLayer::onInitGraph();
    pipeGraph->addNode(lowLayer->getLayer());
}

void VkHighPassLayer::onInitNode() {
    lowLayer->addLine(this, 0, 1);
    setStartNode(this, 0);
    setStartNode(lowLayer.get(), 1);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce