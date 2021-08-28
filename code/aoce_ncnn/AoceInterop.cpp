#include "AoceInterop.hpp"

#include "aoce_vulkan/vulkan/VulkanBuffer.hpp"

namespace aoce {

AoceInterop::AoceInterop(/* args */) {}

AoceInterop::~AoceInterop() {}

void AoceInterop::getTest(ncnn::Net* net) {
    const ncnn::VulkanDevice* device = net->vulkan_device();
    ncnn::VkCompute cmd(device);

    vulkan::VulkanBuffer* aBuffer = nullptr;
    ncnn::VkBufferMemory nBuffer = {};
    nBuffer.buffer = aBuffer->buffer;
    nBuffer.memory = aBuffer->memory;
    nBuffer.offset = 0;
    nBuffer.capacity = aBuffer->getBufferSize();
    nBuffer.mapped_ptr = aBuffer->getCpuData();
    nBuffer.access_flags = 0;
    nBuffer.stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    nBuffer.refcount = 1;

    ncnn::VkAllocator* vkallocator = device->acquire_blob_allocator();
    ncnn::VkMat nMat(320, 240, 3, &nBuffer, 4, vkallocator);
    // nMat.create(320, 240, 3, 1, device->acquire_blob_allocator());
}

}  // namespace aoce