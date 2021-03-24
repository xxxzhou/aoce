#include "VkTransposeLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkTransposeLayer::VkTransposeLayer(/* args */) {
    setUBOSize(sizeof(TransposeParamet), true);
    onUpdateParamet();
}

VkTransposeLayer::~VkTransposeLayer() {}

void VkTransposeLayer::onInitGraph() {
    std::string path = "glsl/transpose.comp.spv";
    shader->loadShaderModule(context->device, path);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkTransposeLayer::onInitLayer() {
    outFormats[0].width = inFormats[0].height;
    outFormats[0].height = inFormats[0].width;
    sizeX = divUp(outFormats[0].width, groupX);
    sizeY = divUp(outFormats[0].height, groupY);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce