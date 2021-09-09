#include "VkCommand.hpp"

#include "vulkan/VulkanManager.hpp"

namespace aoce {
namespace vulkan {

VkCommand::VkCommand(/* args */) {
    this->device = VulkanManager::Get().device;
    this->cmdPool = VulkanManager::Get().cmdPool;
    // command buffer
    VkCommandBufferAllocateInfo cmdBufInfo = {};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufInfo.commandPool = cmdPool;
    cmdBufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(
        vkAllocateCommandBuffers(device, &cmdBufInfo, &computerCmd));
}

VkCommand::~VkCommand() {}

}  // namespace vulkan
}  // namespace aoce