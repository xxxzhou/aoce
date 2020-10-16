#include "VkInputLayer.hpp"
namespace aoce {
namespace vulkan {
namespace layer {

VkInputLayer::VkInputLayer(/* args */) { bInput = true; }

VkInputLayer::~VkInputLayer() {}

void VkInputLayer::onSetImage(VideoFormat videoFormat, int32_t index) {
    assert(index < inCount);
    // 根据各种格式调整
    inFormats[0] = videoFormat2ImageFormat(videoFormat);
    this->videoFormat = videoFormat;
}

void VkInputLayer::onInputCpuData(uint8_t* data, int32_t index) {
    assert(index < inCount);
    frameData = data;
}

void VkInputLayer::onInitVkBuffer() {
    int32_t size = inFormats[0].width * inFormats[0].height *
                   getImageTypeSize(inFormats[0].imageType);
    assert(size > 0);
    inBuffer = std::make_unique<VulkanBuffer>();
    inBuffer->initResoure(context, BufferUsage::store, size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, frameData);
}

void VkInputLayer::onPreCmd() {
    outTexs[0]->AddBarrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
    context->BufferToImage(cmd, inBuffer.get(), outTexs[0].get());
}

bool VkInputLayer::onFrame() {
    if (inBuffer) {
        inBuffer->upload(frameData);
        return true;
    }
    return false;
}

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce