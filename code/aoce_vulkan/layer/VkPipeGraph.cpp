#include "VkPipeGraph.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkPipeGraph::VkPipeGraph(/* args */) {
    context = std::make_unique<VulkanContext>();
    context->InitContext();

    // 创建cpu-gpu通知
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // 默认是有信号
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(context->logicalDevice.device, &fenceInfo, nullptr,
                  &computerFence);
}

VkPipeGraph::~VkPipeGraph() {}

VulkanTexturePtr VkPipeGraph::getOutTex(int32_t node, int32_t outIndex) {
    assert(node < nodes.size());
    VkLayer* vkLayer = static_cast<VkLayer*>(nodes[node]->getLayer());
    assert(outIndex < vkLayer->outputCount);
    return vkLayer->outTexs[outIndex];
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
    return true;
}

bool VkPipeGraph::onRun() {
    // 除开输出层,运行所有层
    for (auto* layer : vkLayers) {
        if (!layer->onFrame()) {
            return false;
        }
    }
    auto device = context->logicalDevice.device;
    vkWaitForFences(device, 1, &computerFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &computerFence);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &context->computerCmd;

    VK_CHECK_RESULT(
        vkQueueSubmit(context->computeQueue, 1, &submitInfo, computerFence));
    // 运行输出层
    for (auto* layer : vkOutputLayers) {
        if (!layer->onFrame()) {
            return false;
        }
    }
    return true;
}
}  // namespace layer
}  // namespace vulkan
}  // namespace aoce