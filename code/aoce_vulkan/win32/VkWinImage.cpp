#include "VkWinImage.hpp"

#include <vulkan/vulkan_win32.h>

#include "../layer/VkHelper.hpp"
#include "../vulkan/VulkanManager.hpp"

using namespace aoce::win;

namespace aoce {
namespace vulkan {

int32_t getMemoryTypeIndex(uint32_t bits) {
    for (uint32_t memoryTypeIndex = 0; (1u << memoryTypeIndex) <= bits;
         memoryTypeIndex++) {
        if ((bits & (1u << memoryTypeIndex)) != 0) {
            return memoryTypeIndex;
        }
    }
    return -1;
}

VkWinImage::VkWinImage(/* args */) {
    vkDevice = VulkanManager::Get().device;
    shardTex = std::make_unique<Dx11SharedTex>();
    tempTex = std::make_unique<Dx11SharedTex>();
#if defined(VK_KHR_external_memory_win32)
    vkGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(vkDevice, "vkGetMemoryWin32HandleKHR"));
    vkGetMemoryWin32HandlePropertiesKHR =
        reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(vkDevice,
                                "vkGetMemoryWin32HandlePropertiesKHR"));
#endif  // defined(VK_KHR_external_memory_win32)
}

VkWinImage::~VkWinImage() { release(); }

void VkWinImage::release() {
    if (vkImage) {
        vkDestroyImage(vkDevice, vkImage, nullptr);
        vkImage = VK_NULL_HANDLE;
    }
    if (memory) {
        vkFreeMemory(vkDevice, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
    bInit = false;
}

void VkWinImage::bindDx11(ID3D11Device* device, ImageFormat format) {
    release();
    this->format = format;
    // 创建一个dx11可以多上下文共享访问的资源
    DXGI_FORMAT dxFormat = getImageDXFormt(format.imageType);
    bInit =
        shardTex->restart(device, format.width, format.height, dxFormat, true);
    bInit &= tempTex->restart(device, format.width, format.height, dxFormat);
    if (!bInit) {
        return;
    }
    VkExternalMemoryHandleTypeFlagBits handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    // 创建对应上面的vulkan资源
    VkExternalMemoryImageCreateInfo externalCreateInfo = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    externalCreateInfo.handleTypes = handleType;
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalCreateInfo;
    imageInfo.flags = 0u;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = layer::ImageFormat2Vk(format.imageType);
    imageInfo.extent = {
        (uint32_t)format.width,
        (uint32_t)format.height,
        1u,
    };
    imageInfo.mipLevels = 1u, imageInfo.arrayLayers = 1u;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK_RESULT(vkCreateImage(vkDevice, &imageInfo, nullptr, &vkImage));

    // 创建从DX11纹理中的ImageMemory,绑定到上面的vkImage.
    const VkImageMemoryRequirementsInfo2 requirementsInfo = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
        nullptr,
        vkImage,
    };
    VkMemoryDedicatedRequirements dedicatedRequirements = {
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
        nullptr,
        VK_FALSE,
        VK_FALSE,
    };
    VkMemoryRequirements2 requirements = {
        VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        &dedicatedRequirements,
        {
            0u,
            0u,
            0u,
        },
    };
    vkGetImageMemoryRequirements2(vkDevice, &requirementsInfo, &requirements);
    // memoryTypeIndex
    VkMemoryWin32HandlePropertiesKHR memoryWin32HandleProperties = {
        VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR, nullptr, 0u};
    vkGetMemoryWin32HandlePropertiesKHR(vkDevice, handleType,
                                        shardTex->sharedHandle,
                                        &memoryWin32HandleProperties);
    VkMemoryRequirements memReq = requirements.memoryRequirements;
    // 后面需要搞清楚这里的memoryTypeBits具体信息,如何影响VK分配
    uint32_t memoryBit =
        memReq.memoryTypeBits & memoryWin32HandleProperties.memoryTypeBits;
    assert(memoryBit != 0);
    uint32_t memoryTypeIndex = getMemoryTypeIndex(memoryBit);
    assert(memoryTypeIndex >= 0);
    // create image memory
    VkMemoryDedicatedAllocateInfo dedicatedInfo = {
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedInfo.image = vkImage;
    VkImportMemoryWin32HandleInfoKHR importInfo = {
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    if (!!dedicatedRequirements.requiresDedicatedAllocation) {
        importInfo.pNext = &dedicatedInfo;
    }
    importInfo.handleType = handleType;
    importInfo.handle = shardTex->sharedHandle;
    VkMemoryAllocateInfo memoryInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryInfo.pNext = &importInfo;
    memoryInfo.allocationSize = memReq.size;
    memoryInfo.memoryTypeIndex = memoryTypeIndex;
    VK_CHECK_RESULT(vkAllocateMemory(vkDevice, &memoryInfo, nullptr, &memory));
    VkBindImageMemoryInfo BindImageMemoryInfo = {
        VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO};
    BindImageMemoryInfo.image = vkImage;
    BindImageMemoryInfo.memory = memory;
    // VK_CHECK_RESULT(vkBindImageMemory2(vkDevice, 1, &BindImageMemoryInfo));
    VK_CHECK_RESULT(vkBindImageMemory(vkDevice, vkImage, memory, 0));
    bInit = true;
}

void VkWinImage::vkCopyTemp(ID3D11Device* device) {
    if (shardTex && shardTex->texture) {
        copyTextureToShared(device, tempTex->sharedHandle,
                            shardTex->texture->texture);
        tempTex->bGpuUpdate = true;
    }
}

void VkWinImage::tempCopyDx11(ID3D11Device* device,ID3D11Texture2D* dx11Tex){
    if(tempTex->bGpuUpdate){
        copySharedToTexture(device, tempTex->sharedHandle, dx11Tex);
        tempTex->bGpuUpdate = false;
    }
}

}  // namespace vulkan
}  // namespace aoce