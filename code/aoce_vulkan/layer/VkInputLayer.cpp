#include "VkInputLayer.hpp"
namespace aoce {
namespace vk {
namespace layer {

VkInputLayer::VkInputLayer(/* args */) { bInput = true; }

VkInputLayer::~VkInputLayer() {}

void VkInputLayer::setImage(ImageFormat imageFormat, int32_t index) {}

void VkInputLayer::onInitLayer() {}

}  // namespace layer
}  // namespace vk
}  // namespace aoce