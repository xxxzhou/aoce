#include "VkChromKeyLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkChromKeyLayer::VkChromKeyLayer(/* args */) {
    setUBOSize(sizeof(ChromKeyParamet), true);
}

VkChromKeyLayer::~VkChromKeyLayer() {} 

void VkChromKeyLayer::onInitGraph() {
    std::string path = "glsl/chromaKey.comp.spv";
    shader->loadShaderModule(context->device, path);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce