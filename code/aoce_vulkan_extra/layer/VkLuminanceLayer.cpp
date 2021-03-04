#include "VkLuminanceLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkLuminanceLayer::VkLuminanceLayer(/* args */) {}

VkLuminanceLayer::~VkLuminanceLayer() {}

void VkLuminanceLayer::onInitGraph() {
    std::string path = "glsl/luminance.comp.spv";
    shader->loadShaderModule(context->device, path);
    std::vector<UBOLayoutItem> items = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkLuminanceLayer::onInitLayer() {
    inFormats[0].imageType = ImageType::rgba8;
    outFormats[0].imageType = ImageType::r8;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce