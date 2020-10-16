#include "VkOutputLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

VkOutputLayer::VkOutputLayer(/* args */) { bOutput = true; }

VkOutputLayer::~VkOutputLayer() {}

void VkOutputLayer::onInitVkBuffer() {
    int32_t size = outFormats[0].width * outFormats[0].height *
                   getImageTypeSize(outFormats[0].imageType);
    assert(size > 0);
    outBuffer = std::make_unique<VulkanBuffer>();
    outBuffer->initResoure(context, BufferUsage::store, size,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    cpuData.resize(size);
}

void VkOutputLayer::onPreCmd() {
    inTexs[0]->AddBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
    context->ImageToBuffer(cmd, inTexs[0].get(), outBuffer.get());
}

bool VkOutputLayer::onFrame() {
    // outBuffer->upload(cpuData.data());
    onImageProcessHandle(outBuffer->getCpuData(), outFormats[0].width,
                         outFormats[0].height, 0);
    return true;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce