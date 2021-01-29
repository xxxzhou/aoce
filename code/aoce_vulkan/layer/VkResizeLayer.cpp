#include "VkResizeLayer.hpp"

#include "VkPipeGraph.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

struct VkResizeParamet {
    int32_t bLinear = 1;
    float fx = 1.f;
    float fy = 1.f;
};

VkResizeLayer::VkResizeLayer(/* args */) {
    setUBOSize(sizeof(VkResizeParamet));
}

VkResizeLayer::~VkResizeLayer() {}

void VkResizeLayer::onUpdateParamet() {
    if (pipeGraph) {
        pipeGraph->reset();
    }
}

void VkResizeLayer::onInitGraph() {
    std::string path = "glsl/resize.comp.spv";
    shader->loadShaderModule(context->device, path);
    // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkResizeLayer::onInitLayer() {
    assert(paramet.newWidth > 0 && paramet.newHeight > 0);
    outFormats[0].width = paramet.newWidth;
    outFormats[0].height = paramet.newHeight;
    VkResizeParamet vpar = {};
    vpar.bLinear = paramet.bLinear;
    vpar.fx = (float)inFormats[0].width / outFormats[0].width;
    vpar.fy = (float)inFormats[0].height / outFormats[0].height;
    memcpy(constBufCpu.data(), &vpar, conBufSize);

    sizeX = divUp(outFormats[0].width, groupX);
    sizeY = divUp(outFormats[0].height, groupY);
}

void VkResizeLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (paramet.bLinear) {
        inTexs[0]->descInfo.sampler = context->linearSampler;
    } else {
        inTexs[0]->descInfo.sampler = context->nearestSampler;
    }
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &outTexs[0]->descInfo,
                            &constBuf->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce