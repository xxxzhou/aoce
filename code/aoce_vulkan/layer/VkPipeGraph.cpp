#include "VkPipeGraph.hpp"

#include <thread>

namespace aoce {
namespace vulkan {
namespace layer {

VkPipeGraph::VkPipeGraph(/* args */) {
    // delayGpu = true;

    context = std::make_unique<VulkanContext>();
    context->initContext();

    // 创建cpu-gpu通知
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // 默认是有信号,通过运行的第一桢
    if (delayGpu) {
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }
    vkCreateFence(context->device, &fenceInfo, nullptr, &computerFence);

    // 创建一个Event
    VkEventCreateInfo eventInfo = {};
    eventInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventInfo.pNext = nullptr;
    eventInfo.flags = 0;
    vkCreateEvent(context->device, &eventInfo, nullptr, &outEvent);
    stageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;

    gpu = GpuType::vulkan;
}

VkPipeGraph::~VkPipeGraph() {
    if (outEvent) {
        vkDestroyEvent(context->device, outEvent, VK_NULL_HANDLE);
        outEvent = VK_NULL_HANDLE;
    }
    if (computerFence) {
        vkDestroyFence(context->device, computerFence, nullptr);
        computerFence = VK_NULL_HANDLE;
    }
}

VulkanTexturePtr VkPipeGraph::getOutTex(int32_t node, int32_t outIndex) {
    assert(node < nodes.size());
    VkLayer* vkLayer = static_cast<VkLayer*>(nodes[node]->getLayer());
    assert(outIndex < vkLayer->outCount);
    return vkLayer->outTexs[outIndex];
}

bool VkPipeGraph::resourceReady(){
    // 资源是否已经重新生成
    auto res = vkGetEventStatus(context->device, outEvent);
    return res == VK_EVENT_SET;
}

void VkPipeGraph::onReset() {
    // while (vkGetFenceStatus(context->device, computerFence) == VK_NOT_READY)
    // {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // }
    // auto res = vkGetEventStatus(context->device, outEvent);
    // assert(res != VK_EVENT_RESET);
    // 告诉别的线程,需要等待资源重新生成
    vkResetEvent(context->device, outEvent);
    vkResetCommandBuffer(context->computerCmd,
                         VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    vkQueueWaitIdle(context->computeQueue);
}

bool VkPipeGraph::onInitBuffers() {
    // 填充CommandBuffer
    VkCommandBuffer cmd = context->computerCmd;
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    // 拿到所有层
    vkOutputLayers.clear();
    vkLayers.clear();
    for (auto index : nodeExcs) {
        VkLayer* vkLayer = static_cast<VkLayer*>(nodes[index]->getLayer());
        vkLayer->onPreCmd();
        if (vkLayer->bOutput) {
            vkOutputLayers.push_back(vkLayer);
        } else {
            vkLayers.push_back(vkLayer);
        }
    }
    // 记录状态改为可执行状态
    vkEndCommandBuffer(cmd);
    // 告诉别的线程,资源已经准备好
    vkSetEvent(context->device, outEvent);
    return true;
}

bool VkPipeGraph::onRun() {
    // vkCmdWaitEvents(context->computerCmd, 1, &outEvent, stageFlags,
    //                 stageFlags, 0, nullptr, 0, nullptr, 0,
    //                 nullptr);
    // vkCmdResetEvent(context->computerCmd, outEvent, stageFlags);
    // 更新所有层的参数
    for (auto* layer : vkLayers) {
        if (layer->bParametChange) {
            layer->updateUBO();
            layer->bParametChange = false;
        }
    }
    // 除开输出层,运行所有层
    for (auto* layer : vkLayers) {
        if (!layer->onFrame()) {
            return false;
        }
    }
    // 等待上一桢执行完成,这种模式会导致当前桢输出的是上一桢的数据
    // 这样不需要等待当前GPU运行这桢数据完成.
    if (delayGpu) {
        vkWaitForFences(context->device, 1, &computerFence, VK_TRUE,
                        UINT64_MAX);  // UINT64_MAX
        vkResetFences(context->device, 1, &computerFence);
        // 运行输出层
        for (auto* layer : vkOutputLayers) {
            if (!layer->onFrame()) {
                return false;
            }
        }
    }
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &context->computerCmd;
    VK_CHECK_RESULT(
        vkQueueSubmit(context->computeQueue, 1, &submitInfo, computerFence));
    if (!delayGpu) {
        // 运行输出层
        for (auto* layer : vkOutputLayers) {
            if (!layer->onFrame()) {
                return false;
            }
        }
        vkWaitForFences(context->device, 1, &computerFence, VK_TRUE,
                        UINT64_MAX);  // UINT64_MAX
        vkResetFences(context->device, 1, &computerFence);
    }
    // vkCmdSetEvent(context->computerCmd, outEvent, stageFlags);
    return true;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce