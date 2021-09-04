#include "VkDrawLayer.hpp"

#include "aoce/layer/PipeGraph.hpp"
#if __ANDROID__
#include "aoce_vulkan/android/vulkan_wrapper.h"
#endif

namespace aoce {
namespace vulkan {
namespace layer {

VkDrawPointsPreLayer::VkDrawPointsPreLayer(/* args */) {
    bInput = true;
    glslPath = "glsl/drawPoints.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkDrawPointsPreLayer::~VkDrawPointsPreLayer() {}

void VkDrawPointsPreLayer::setImageFormat(const ImageFormat& imageFormat) {
    inFormats[0] = imageFormat;
    outFormats[0] = imageFormat;
}

void VkDrawPointsPreLayer::drawPoints(const vec2* points, int32_t size,
                                      vec4 color, int32_t raduis) {
    paramet.showCount = std::min(maxPoint, size);
    paramet.radius = raduis;
    paramet.color = color;
    updateUBO(&paramet);
    bParametChange = true;
    if (inBuffer) {
        memcpy(inBuffer->getCpuData(), points, paramet.showCount * 2 * 4);
    }
}

void VkDrawPointsPreLayer::onInitGraph() {
    shader->loadShaderModule(context->device, glslPath);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();

    inFormats[0].imageType = ImageType::rgba8;
    inFormats[0].width = 640;
    inFormats[0].height = 320;
}

void VkDrawPointsPreLayer::onInitVkBuffer() {
    inBuffer = std::make_unique<VulkanBuffer>();
    inBuffer->initResoure(BufferUsage::store, maxPoint * 4 * 2,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    sizeX = divUp(maxPoint, 240);
    sizeY = 1;
}

void VkDrawPointsPreLayer::onInitPipe() {
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->updateSetLayout(0, 0, &inBuffer->descInfo, &outTexs[0]->descInfo,
                            &constBuf->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

void VkDrawPointsPreLayer::onCommand() {
    clearColor({0, 0, 0, 0});

    outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
}

VkDrawPointsLayer::VkDrawPointsLayer(/* args */) {
    preLayer = std::make_unique<VkDrawPointsPreLayer>();

    glslPath = "glsl/drawPointBlend.comp.spv";
    inCount = 2;
    onUpdateParamet();
}

VkDrawPointsLayer::~VkDrawPointsLayer() {}

void VkDrawPointsLayer::onUpdateParamet() { preLayer->updateParamet(paramet); }

void VkDrawPointsLayer::onInitNode() {
    pipeGraph->addNode(preLayer->getLayer());
    preLayer->addLine(this, 0, 1);
}

void VkDrawPointsLayer::onInitLayer() {
    // 混合大小以及VkDrawPointsPreLayer里大小以第一个输入为主
    inFormats[1] = inFormats[0];
    preLayer->setImageFormat(inFormats[0]);
    VkLayer::onInitLayer();
}

// void VkDrawPointsLayer::onCommand() {
//     VkLayer::onCommand();
// }

void VkDrawPointsLayer::drawPoints(const vec2* points, int32_t size, vec4 color,
                                   int32_t raduis) {
    preLayer->drawPoints(points, size, color, raduis);
}

VkDrawRectLayer::VkDrawRectLayer(/* args */) {
    glslPath = "glsl/drawRect.comp.spv";
    setUBOSize(sizeof(paramet), true);
    updateUBO(&paramet);
}

VkDrawRectLayer::~VkDrawRectLayer() {}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce