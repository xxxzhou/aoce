#include "VkSeparableLinearLayer.hpp"

#define PATCH_PER_BLOCK 4

namespace aoce {
namespace vulkan {
namespace layer {

VkSeparableLinearLayer::VkSeparableLinearLayer(/* args */) {}

VkSeparableLinearLayer::~VkSeparableLinearLayer() {}

void VkSeparableLinearLayer::onInitGraph() {
    std::string path = "glsl/filterRow.comp.spv";
    std::string path2 = "glsl/filterColumn.comp.spv";
    shader->loadShaderModule(context->device, path);

    shader2 = std::make_unique<VulkanShader>();
    shader2->loadShaderModule(context->device, path);

    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkSeparableLinearLayer::onInitLayer() {
    // row[rSizeX,sizeY] column[sizeX,cSizeY]
    sizeX = divUp(inFormats[0].width, groupX);
    sizeY = divUp(inFormats[0].height, groupY);
    // 线程组只取平常宽的1/4
    rSizeX = divUp(inFormats[0].width, groupX * PATCH_PER_BLOCK);
    cSizeY = divUp(inFormats[0].height, groupY* PATCH_PER_BLOCK);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce