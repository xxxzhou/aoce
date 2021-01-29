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

VkLayer::~VkLayer() {}

void VkLayer::setUBOSize(int size, bool bMatchParamet) {
    conBufSize = size;
    constBufCpu.resize(conBufSize);
    bParametMatch = bMatchParamet;
}

void VkLayer::updateUBO() {
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
    // for(int i =0;i<inCount;i++){
    //     inFormats[i].imageType = ImageType::rgba8;
    // }
    // for(int i =0;i<outCount;i++){
    //     outFormats[i].imageType = ImageType::rgba8;
    // }
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
        texPtr->InitResource(format.width, format.height, vkft,
                             VK_IMAGE_USAGE_STORAGE_BIT, 0);
        outTexs.push_back(texPtr);
    }
}

void VkLayer::onInitBuffer() {
    if (!bInput) {
        inTexs.clear();
        for (int32_t i = 0; i < inCount; i++) {
            auto& inLayer = this->inLayers[i];
            inTexs.push_back(
                vkPipeGraph->getOutTex(inLayer.nodeIndex, inLayer.outputIndex));
        }
    }
    if (!bOutput) {
        createOutTexs();
    }
    onInitVkBuffer();
    onInitPipe();
    // 默认更新一次UBO
    updateUBO();
}

bool VkLayer::onFrame() { return true; }

void VkLayer::onInitPipe() {
    inTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outTexs[0]->descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    layout->updateSetLayout(0, 0, &inTexs[0]->descInfo, &outTexs[0]->descInfo,
                            &constBuf->descInfo);
    auto computePipelineInfo = VulkanPipeline::createComputePipelineInfo(
        layout->pipelineLayout, shader->shaderStage);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        context->device, context->pipelineCache, 1, &computePipelineInfo,
        nullptr, &computerPipeline));
}

void VkLayer::onPreCmd() {
    inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    outTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computerPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout->pipelineLayout, 0, 1,
                            layout->descSets[0].data(), 0, 0);
    vkCmdDispatch(cmd, sizeX, sizeY, 1);
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce