#include "VkLayer.hpp"

#include "../vulkan/VulkanPipeline.hpp"
#include "VkPipeGraph.hpp"

#ifdef _WIN32
#include <windows.h>
#elif __ANDROID__
#include <android/asset_manager.h>
#endif

namespace aoce {
namespace vulkan {
namespace layer {

VkLayer::VkLayer(/* args */) { gpu = GpuType::vulkan; }

VkLayer::~VkLayer() { constBufCpu.clear(); }

void VkLayer::setUBOSize(int size, bool bMatchParamet) {
    conBufSize = size;
    constBufCpu.resize(conBufSize);
    bParametMatch = bMatchParamet;
}

void VkLayer::generateLayout() {
    if (layout->pipelineLayout != VK_NULL_HANDLE) {
        return;
    }
    std::vector<UBOLayoutItem> items;
    for (int i = 0; i < inCount; i++) {
        VkDescriptorType vdt = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (getSampled(i)) {
            vdt = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        items.push_back({vdt, VK_SHADER_STAGE_COMPUTE_BIT});
    }
    for (int i = 0; i < outCount; i++) {
        items.push_back(
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT});
    }
    if (constBuf) {
        items.push_back(
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT});
    }
    layout->addSetLayout(items);
    layout->generateLayout();
}

void VkLayer::updateUBO(void* data) {
    memcpy(constBufCpu.data(), data, conBufSize);
}

void VkLayer::submitUBO() {
    if (constBuf) {
        constBuf->upload(constBufCpu.data());
        constBuf->submit();
    }
}

void VkLayer::onInit() {
    BaseLayer::onInit();
    vkPipeGraph =
        static_cast<VkPipeGraph*>(pipeGraph);  // dynamic_cast android open rtti
    context = vkPipeGraph->getContext();
    assert(context != nullptr);
    cmd = context->computerCmd;
    if (!bInput) {
        inTexs.resize(inCount);
    }
    outTexs.resize(outCount);
    layout = std::make_unique<UBOLayout>();
    shader = std::make_unique<VulkanShader>();
    // 是否需要UBO
    if (conBufSize > 0) {
        constBuf = std::make_unique<VulkanBuffer>();
        constBuf->initResoure(BufferUsage::store, conBufSize,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              constBufCpu.data());
    }
    onInitGraph();
}

void VkLayer::onInitLayer() {
    if (inCount > 0) {
        sizeX = divUp(inFormats[0].width, groupX);
        sizeY = divUp(inFormats[0].height, groupY);
    }
}

void VkLayer::createOutTexs() {
    outTexs.clear();
    for (int32_t i = 0; i < outCount; i++) {
        const ImageFormat& format = outFormats[i];
        VkFormat vkft = ImageFormat2Vk(format.imageType);
        VulkanTexturePtr texPtr(new VulkanTexture());
        // VkMemoryPropertyFlags
        VkMemoryPropertyFlags texFlags = VK_IMAGE_USAGE_STORAGE_BIT;
        auto& outLayer = this->outLayers[i];
        // 需要检测当前层的输出层是否需要当前层的纹理需要采样
        bool bMustSampled = false;
        // 是否需要输出
        bool bMustOutput = false;
        for (int32_t i = 0; i < outLayer.size(); i++) {
            if (vkPipeGraph->getMustSampled(outLayer[i].nodeIndex,
                                            outLayer[i].siteIndex)) {
                bMustSampled = true;
            }
            if (vkPipeGraph->bOutLayer(outLayer[i].nodeIndex)) {
                bMustOutput = true;
            };
        }
        // 查看对应输出层是否需要采样功能
        if (bMustSampled) {
            texFlags = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        // 输入与输出都给传输位
        if (bInput || bMustOutput) {
            texFlags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            texFlags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }
        texPtr->InitResource(format.width, format.height, vkft, texFlags, 0);
        outTexs.push_back(texPtr);
    }
}

void VkLayer::clearColor(vec4 color) {
    for (int i = 0; i < outCount; i++) {
        outTexs[i]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT);
        VkImageSubresourceRange subResourceRange = {};
        subResourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subResourceRange.baseMipLevel = 0;
        subResourceRange.levelCount = 1;
        subResourceRange.baseArrayLayer = 0;
        subResourceRange.layerCount = 1;

        VkClearColorValue clearColor = {color.x, color.y, color.z, color.w};
        vkCmdClearColorImage(cmd, outTexs[i]->image, VK_IMAGE_LAYOUT_GENERAL,
                             &clearColor, 1, &subResourceRange);
    }
}

void VkLayer::onInitBuffer() {
    if (!bInput) {
        inTexs.clear();
        for (int32_t i = 0; i < inCount; i++) {
            auto& inLayer = this->inLayers[i];
            inTexs.push_back(
                vkPipeGraph->getOutTex(inLayer.nodeIndex, inLayer.siteIndex));
        }
    }
    if (!bOutput) {
        createOutTexs();
    }
    onInitVkBuffer();
    onInitPipe();
    // 默认更新一次UBO
    submitUBO();
}

bool VkLayer::onFrame() { return true; }

void VkLayer::onInitGraph() {
    if (!glslPath.empty()) {
        shader->loadShaderModule(context->device, glslPath);
    }
    generateLayout();
}

void VkLayer::onInitPipe() {
    std::vector<void*> bufferInfos;
    for (int i = 0; i < inCount; i++) {
        inTexs[i]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        // 需要采样就提供,否则不提供
        if (getSampled(i)) {
            VkSampler sampler = context->linearSampler;
            if (sampledNearest(i)) {
                sampler = context->nearestSampler;
            }
            inTexs[i]->descInfo.sampler = sampler;
        }
        bufferInfos.push_back(&inTexs[i]->descInfo);
    }
    for (int i = 0; i < outCount; i++) {
        outTexs[i]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        bufferInfos.push_back(&outTexs[i]->descInfo);
    }
    if (constBuf) {
        bufferInfos.push_back(&constBuf->descInfo);
    }
    layout->updateSetLayout(0, 0, bufferInfos);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

void VkLayer::onPreFrame() {
    if (bParametChange) {
        submitUBO();
        bParametChange = false;
    }
}

void VkLayer::onPreCmd() {
    for (int i = 0; i < inCount; i++) {
        inTexs[i]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
    }
    for (int i = 0; i < outCount; i++) {
        outTexs[i]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce