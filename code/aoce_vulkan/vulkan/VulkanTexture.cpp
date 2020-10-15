#include "VulkanTexture.hpp"

#include "VulkanContext.hpp"
#include "VulkanHelper.hpp"

namespace aoce {
namespace vulkan {

VulkanTexture::VulkanTexture() {}
VulkanTexture::~VulkanTexture() {
    if (sampler) {
        vkDestroySampler(device, sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }
    if (view) {
        vkDestroyImageView(device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (image) {
        vkDestroyImage(device, image, nullptr);
        image = VK_NULL_HANDLE;
    }
    if (memory) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

void VulkanTexture::InitResource(class VulkanContext *context, uint32_t width,
                                 uint32_t height, VkFormat format,
                                 VkImageUsageFlags usageFlag,
                                 VkMemoryPropertyFlags memoryFlag,
                                 uint8_t *cpuData, uint8_t cpuPitch) {
    this->device = context->logicalDevice.device;
    bool bGpu = cpuData == nullptr;
    this->width = width;
    this->height = height;
    this->format = format;
    // 重新初始化
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
    accessFlags = 0;
    if (!bGpu) {
        // 初始化只能UNDEFINED/PREINITIALIZED,PREINITIALIZED可以用内存数据初始化
        layout = VK_IMAGE_LAYOUT_PREINITIALIZED;
        accessFlags = VK_ACCESS_HOST_WRITE_BIT;
    }
    stageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    // SRV,UVA资源一般不支持LINEAR格式,LINEAR限制较大,请先检查是否支持相应UsageFlag
    VkImageTiling tiling =
        bGpu ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = 0;
    imageInfo.pNext = nullptr;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;  //一般VK_FORMAT_R8G8B8A8_UNORM
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = tiling;
    // VK_IMAGE_LAYOUT_PREINITIALIZED 内存数据初始化,可以直接存储在设备内存
    imageInfo.initialLayout = layout;
    imageInfo.usage = usageFlag;  // usageFlag;
    // 一般来说,我们只需要单队列访问 VK_SHARING_MODE_EXCLUSIVE,故为0
    imageInfo.queueFamilyIndexCount = 0;
    imageInfo.pQueueFamilyIndices = nullptr;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // 创建image
    VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &image));
    VkMemoryRequirements requires;
    vkGetImageMemoryRequirements(device, image, &requires);
    uint32_t memoryTypeIndex = 0;
    bool getIndex =
        getMemoryTypeIndex(context->physicalDevice, requires.memoryTypeBits,
                           memoryFlag, memoryTypeIndex);
    assert(getIndex == true);
    VkMemoryAllocateInfo memoryInfo = {};
    memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryInfo.pNext = nullptr;
    memoryInfo.memoryTypeIndex = memoryTypeIndex;
    memoryInfo.allocationSize = requires.size;
    // 生成设备显存空间
    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryInfo, nullptr, &memory));
    VK_CHECK_RESULT(vkBindImageMemory(device, image, memory, 0));
    //直接传入cpu数据到image,需要检查image相应的rowPitch
    //可以考虑使用buffer,然后vkCmdCopyBufferToImage
    if (cpuData) {
        uint8_t *pData;
        VkImageSubresource subres = {};
        subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subres.mipLevel = 0;
        subres.arrayLayer = 0;
        VkSubresourceLayout layout = {};
        vkGetImageSubresourceLayout(device, image, &subres, &layout);
        VK_CHECK_RESULT(
            vkMapMemory(device, memory, 0, requires.size, 0, (void **)&pData));
        // uint8_t cpuPitch = width * getByteSize(format);
        if (cpuPitch != 0 && cpuPitch != layout.rowPitch) {
            assert(layout.rowPitch >= cpuPitch);
            for (uint32_t i = 0; i < height; i++) {
                memcpy(pData + layout.rowPitch * i, cpuData + i * cpuPitch,
                       cpuPitch);
            }
        } else {
            memcpy(pData, cpuData, requires.size);
        }
        // 这段逻辑需要再考虑下
        if ((memoryFlag & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange memRange;
            memRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            memRange.pNext = nullptr;
            memRange.memory = memory;
            memRange.offset = 0;
            memRange.size = requires.size;
            VK_CHECK_RESULT(vkFlushMappedMemoryRanges(device, 1, &memRange));
        }
        vkUnmapMemory(device, memory);
    }
    // 创建sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.maxAnisotropy = 1;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0;
    samplerCreateInfo.maxLod = 0.0;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(
        vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler));
    // 创建view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &view));
    // 填充VkDescriptorImageInfo
    descInfo.imageView = view;
    descInfo.sampler = sampler;
    // 渲染管线一般用VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,计算管线用general
    descInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

void VulkanTexture::AddBarrier(VkCommandBuffer command, VkImageLayout newLayout,
                               VkPipelineStageFlags newStageFlags,
                               VkAccessFlags newAccessFlags) {
    VkImageLayout oldLayout = layout;
    VkPipelineStageFlags oldStageFlags = stageFlags;

    changeLayout(command, image, oldLayout, newLayout, oldStageFlags,
                 newStageFlags, VK_IMAGE_ASPECT_COLOR_BIT, newAccessFlags);
    layout = newLayout;
    stageFlags = newStageFlags;
    accessFlags = newAccessFlags;
}

}  // namespace vulkan
}  // namespace aoce
