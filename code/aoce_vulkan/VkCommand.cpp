#include "VkCommand.hpp"

#include "vulkan/VulkanManager.hpp"

namespace aoce {
namespace vulkan {

VkCommand::VkCommand(/* args */) {
    device = VulkanManager::Get().device;
    queue = VulkanManager::Get().computeQueue;
    cmdPool = VulkanManager::Get().cmdPool;
    // command buffer
    VkCommandBufferAllocateInfo cmdBufInfo = {};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufInfo.commandPool = cmdPool;
    cmdBufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufInfo, &cmd));
    // 创建cpu-gpu通知
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // 初始是没有信号的
    fenceInfo.flags = 0;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);
    // 设置为记录状态
    beginRecord();
}

VkCommand::~VkCommand() {
    if (fence) {
        vkDestroyFence(device, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
    if (cmd) {
        vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
        cmd = VK_NULL_HANDLE;
    }
}

void VkCommand::beginRecord() {
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 只会被执行一次,然后被销毁.
    // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT 可以多次执行或暂停
    // 把flags设置为0是安全的,能多次执行
    cmdBeginInfo.flags = 0;
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    bRecord = true;
    bSubmit = false;
}

void VkCommand::endRecord() {
    // 记录状态改为可执行状态
    vkEndCommandBuffer(cmd);
    bRecord = false;
}

void VkCommand::barrier(VkBuffer buffer, VkPipelineStageFlags stageFlag,
                        VkAccessFlags assessFlag,
                        VkPipelineStageFlags oldStageFlag,
                        VkAccessFlags oldAssessFlag) {
    VkBufferMemoryBarrier bufBarrier = {};
    bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;

    bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.buffer = buffer;
    bufBarrier.offset = 0;
    bufBarrier.size = VK_WHOLE_SIZE;
    bufBarrier.srcAccessMask = oldAssessFlag;
    bufBarrier.dstAccessMask = assessFlag;
    vkCmdPipelineBarrier(cmd, oldStageFlag, stageFlag, 0, 0, nullptr,
                         1, &bufBarrier, 0, nullptr);
}

void VkCommand::record(VkBuffer src, VkBuffer dest, int32_t destOffset,
                       int32_t size) {
    VkBufferCopy region = {};
    region.srcOffset = 0;
    region.dstOffset = destOffset;
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dest, 1, &region);
}

void VkCommand::submit() {
    if (!bSubmit) {
        endRecord();
        bSubmit = true;
    }
    // 提交GPU执行
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    // 等待GPU执行完成给出信号
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    // 重置信号
    vkResetFences(device, 1, &fence);
}

void VkCommand::reset() {
    // 重置CMD状态
    vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    // 重新开始记录
    beginRecord();
}

}  // namespace vulkan
}  // namespace aoce