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
    // 如果是延迟模式,默认是有信号,通过运行的第一桢
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
#if WIN32
    win::createDevice11(&device, &ctx);
#endif
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

#if WIN32
ID3D11Device* VkPipeGraph::getD3D11Device() { return device; }
void VkPipeGraph::addOutMemory(VkWinImage* winImage) {
    winImages.push_back(winImage);
    // outMemorys.push_back(winImage->getDeviceMemory());
}
#endif

VulkanTexturePtr VkPipeGraph::getOutTex(int32_t node, int32_t outIndex) {
    assert(node < nodes.size());
    VkLayer* vkLayer = static_cast<VkLayer*>(nodes[node]->getLayer());
    assert(outIndex < vkLayer->outCount);
    return vkLayer->outTexs[outIndex];
}

bool VkPipeGraph::getMustSampled(int32_t node, int32_t inIndex) {
    assert(node < nodes.size());
    VkLayer* vkLayer = static_cast<VkLayer*>(nodes[node]->getLayer());
    assert(inIndex < vkLayer->inCount);
    return vkLayer->getSampled(inIndex);
}

bool VkPipeGraph::bOutLayer(int32_t node) {
    assert(node < nodes.size());
    VkLayer* vkLayer = static_cast<VkLayer*>(nodes[node]->getLayer());
    return vkLayer->bOutput;
}

bool VkPipeGraph::resourceReady() {
    // 资源是否已经重新生成
    auto res = vkGetEventStatus(context->device, outEvent);
    return res == VK_EVENT_SET;
}

void VkPipeGraph::onReset() {
// 告诉别的线程,需要等待资源重新生成
#if WIN32
    // outMemorys.clear();
    winImages.clear();
#endif
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

bool VkPipeGraph::executeOut() {
    // 等等信号
    vkWaitForFences(context->device, 1, &computerFence, VK_TRUE,
                    UINT64_MAX);  // UINT64_MAX
#if WIN32
    for (auto* winImage : winImages) {
        winImage->vkCopyTemp(device);
    }
#endif
    // 重置无信号
    vkResetFences(context->device, 1, &computerFence);
    // 运行输出层
    for (auto* layer : vkOutputLayers) {
        if (!layer->onFrame()) {
            return false;
        }
    }
    return true;
}

bool VkPipeGraph::onRun() {
    // 更新所有层的发生改动的vulkan资源
    for (auto* layer : vkLayers) {
        layer->onPreFrame();
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
        executeOut();
    }
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &context->computerCmd;
    VK_CHECK_RESULT(
        vkQueueSubmit(context->computeQueue, 1, &submitInfo, computerFence));
    // 同步当前桢完成并输出
    if (!delayGpu) {
        executeOut();
    }
    return true;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce

// win dx11/vulkan keymutex
//#if WIN32
// if (outMemorys.size() > 0) {
//     uint64_t write = AOCE_DX11_MUTEX_WRITE;
//     uint64_t read = AOCE_DX11_MUTEX_READ;
//     uint32_t timeOut = 0;
//     VkWin32KeyedMutexAcquireReleaseInfoKHR keyedMutexInfo = {};
//     keyedMutexInfo.sType =
//         VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR;
//     keyedMutexInfo.pNext = nullptr;
//     keyedMutexInfo.acquireCount = outMemorys.size();
//     keyedMutexInfo.pAcquireSyncs = outMemorys.data();
//     keyedMutexInfo.pAcquireKeys = &write;
//     keyedMutexInfo.pAcquireTimeouts = &timeOut;
//     keyedMutexInfo.releaseCount = outMemorys.size();
//     keyedMutexInfo.pReleaseSyncs = outMemorys.data();
//     keyedMutexInfo.pReleaseKeys = &read;
//     submitInfo.pNext = &keyedMutexInfo;
// }
//#endif