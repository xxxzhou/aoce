#include "VulkanHelper.hpp"

#include <fstream>
#include <map>

#include "VulkanManager.hpp"
namespace aoce {
namespace vulkan {

// 得到总长,通道数
// https://android.googlesource.com/platform/external/vulkan-validation-layers/+/HEAD/layers/vk_format_utils.cpp
const std::map<VkFormat, FormatInfo> formatTable = {
    {VK_FORMAT_UNDEFINED, {0, 0}},
    {VK_FORMAT_R4G4_UNORM_PACK8, {1, 2}},
    {VK_FORMAT_R4G4B4A4_UNORM_PACK16, {2, 4}},
    {VK_FORMAT_B4G4R4A4_UNORM_PACK16, {2, 4}},
    {VK_FORMAT_R5G6B5_UNORM_PACK16, {2, 3}},
    {VK_FORMAT_B5G6R5_UNORM_PACK16, {2, 3}},
    {VK_FORMAT_R5G5B5A1_UNORM_PACK16, {2, 4}},
    {VK_FORMAT_B5G5R5A1_UNORM_PACK16, {2, 4}},
    {VK_FORMAT_A1R5G5B5_UNORM_PACK16, {2, 4}},
    {VK_FORMAT_R8_UNORM, {1, 1}},
    {VK_FORMAT_R8_SNORM, {1, 1}},
    {VK_FORMAT_R8_USCALED, {1, 1}},
    {VK_FORMAT_R8_SSCALED, {1, 1}},
    {VK_FORMAT_R8_UINT, {1, 1}},
    {VK_FORMAT_R8_SINT, {1, 1}},
    {VK_FORMAT_R8_SRGB, {1, 1}},
    {VK_FORMAT_R8G8_UNORM, {2, 2}},
    {VK_FORMAT_R8G8_SNORM, {2, 2}},
    {VK_FORMAT_R8G8_USCALED, {2, 2}},
    {VK_FORMAT_R8G8_SSCALED, {2, 2}},
    {VK_FORMAT_R8G8_UINT, {2, 2}},
    {VK_FORMAT_R8G8_SINT, {2, 2}},
    {VK_FORMAT_R8G8_SRGB, {2, 2}},
    {VK_FORMAT_R8G8B8_UNORM, {3, 3}},
    {VK_FORMAT_R8G8B8_SNORM, {3, 3}},
    {VK_FORMAT_R8G8B8_USCALED, {3, 3}},
    {VK_FORMAT_R8G8B8_SSCALED, {3, 3}},
    {VK_FORMAT_R8G8B8_UINT, {3, 3}},
    {VK_FORMAT_R8G8B8_SINT, {3, 3}},
    {VK_FORMAT_R8G8B8_SRGB, {3, 3}},
    {VK_FORMAT_B8G8R8_UNORM, {3, 3}},
    {VK_FORMAT_B8G8R8_SNORM, {3, 3}},
    {VK_FORMAT_B8G8R8_USCALED, {3, 3}},
    {VK_FORMAT_B8G8R8_SSCALED, {3, 3}},
    {VK_FORMAT_B8G8R8_UINT, {3, 3}},
    {VK_FORMAT_B8G8R8_SINT, {3, 3}},
    {VK_FORMAT_B8G8R8_SRGB, {3, 3}},
    {VK_FORMAT_R8G8B8A8_UNORM, {4, 4}},
    {VK_FORMAT_R8G8B8A8_SNORM, {4, 4}},
    {VK_FORMAT_R8G8B8A8_USCALED, {4, 4}},
    {VK_FORMAT_R8G8B8A8_SSCALED, {4, 4}},
    {VK_FORMAT_R8G8B8A8_UINT, {4, 4}},
    {VK_FORMAT_R8G8B8A8_SINT, {4, 4}},
    {VK_FORMAT_R8G8B8A8_SRGB, {4, 4}},
    {VK_FORMAT_B8G8R8A8_UNORM, {4, 4}},
    {VK_FORMAT_B8G8R8A8_SNORM, {4, 4}},
    {VK_FORMAT_B8G8R8A8_USCALED, {4, 4}},
    {VK_FORMAT_B8G8R8A8_SSCALED, {4, 4}},
    {VK_FORMAT_B8G8R8A8_UINT, {4, 4}},
    {VK_FORMAT_B8G8R8A8_SINT, {4, 4}},
    {VK_FORMAT_B8G8R8A8_SRGB, {4, 4}},
    {VK_FORMAT_A8B8G8R8_UNORM_PACK32, {4, 4}},
    {VK_FORMAT_A8B8G8R8_SNORM_PACK32, {4, 4}},
    {VK_FORMAT_A8B8G8R8_USCALED_PACK32, {4, 4}},
    {VK_FORMAT_A8B8G8R8_SSCALED_PACK32, {4, 4}},
    {VK_FORMAT_A8B8G8R8_UINT_PACK32, {4, 4}},
    {VK_FORMAT_A8B8G8R8_SINT_PACK32, {4, 4}},
    {VK_FORMAT_A8B8G8R8_SRGB_PACK32, {4, 4}},
    {VK_FORMAT_A2R10G10B10_UNORM_PACK32, {4, 4}},
    {VK_FORMAT_A2R10G10B10_SNORM_PACK32, {4, 4}},
    {VK_FORMAT_A2R10G10B10_USCALED_PACK32, {4, 4}},
    {VK_FORMAT_A2R10G10B10_SSCALED_PACK32, {4, 4}},
    {VK_FORMAT_A2R10G10B10_UINT_PACK32, {4, 4}},
    {VK_FORMAT_A2R10G10B10_SINT_PACK32, {4, 4}},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, {4, 4}},
    {VK_FORMAT_A2B10G10R10_SNORM_PACK32, {4, 4}},
    {VK_FORMAT_A2B10G10R10_USCALED_PACK32, {4, 4}},
    {VK_FORMAT_A2B10G10R10_SSCALED_PACK32, {4, 4}},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32, {4, 4}},
    {VK_FORMAT_A2B10G10R10_SINT_PACK32, {4, 4}},
    {VK_FORMAT_R16_UNORM, {2, 1}},
    {VK_FORMAT_R16_SNORM, {2, 1}},
    {VK_FORMAT_R16_USCALED, {2, 1}},
    {VK_FORMAT_R16_SSCALED, {2, 1}},
    {VK_FORMAT_R16_UINT, {2, 1}},
    {VK_FORMAT_R16_SINT, {2, 1}},
    {VK_FORMAT_R16_SFLOAT, {2, 1}},
    {VK_FORMAT_R16G16_UNORM, {4, 2}},
    {VK_FORMAT_R16G16_SNORM, {4, 2}},
    {VK_FORMAT_R16G16_USCALED, {4, 2}},
    {VK_FORMAT_R16G16_SSCALED, {4, 2}},
    {VK_FORMAT_R16G16_UINT, {4, 2}},
    {VK_FORMAT_R16G16_SINT, {4, 2}},
    {VK_FORMAT_R16G16_SFLOAT, {4, 2}},
    {VK_FORMAT_R16G16B16_UNORM, {6, 3}},
    {VK_FORMAT_R16G16B16_SNORM, {6, 3}},
    {VK_FORMAT_R16G16B16_USCALED, {6, 3}},
    {VK_FORMAT_R16G16B16_SSCALED, {6, 3}},
    {VK_FORMAT_R16G16B16_UINT, {6, 3}},
    {VK_FORMAT_R16G16B16_SINT, {6, 3}},
    {VK_FORMAT_R16G16B16_SFLOAT, {6, 3}},
    {VK_FORMAT_R16G16B16A16_UNORM, {8, 4}},
    {VK_FORMAT_R16G16B16A16_SNORM, {8, 4}},
    {VK_FORMAT_R16G16B16A16_USCALED, {8, 4}},
    {VK_FORMAT_R16G16B16A16_SSCALED, {8, 4}},
    {VK_FORMAT_R16G16B16A16_UINT, {8, 4}},
    {VK_FORMAT_R16G16B16A16_SINT, {8, 4}},
    {VK_FORMAT_R16G16B16A16_SFLOAT, {8, 4}},
    {VK_FORMAT_R32_UINT, {4, 1}},
    {VK_FORMAT_R32_SINT, {4, 1}},
    {VK_FORMAT_R32_SFLOAT, {4, 1}},
    {VK_FORMAT_R32G32_UINT, {8, 2}},
    {VK_FORMAT_R32G32_SINT, {8, 2}},
    {VK_FORMAT_R32G32_SFLOAT, {8, 2}},
    {VK_FORMAT_R32G32B32_UINT, {12, 3}},
    {VK_FORMAT_R32G32B32_SINT, {12, 3}},
    {VK_FORMAT_R32G32B32_SFLOAT, {12, 3}},
    {VK_FORMAT_R32G32B32A32_UINT, {16, 4}},
    {VK_FORMAT_R32G32B32A32_SINT, {16, 4}},
    {VK_FORMAT_R32G32B32A32_SFLOAT, {16, 4}},
    {VK_FORMAT_R64_UINT, {8, 1}},
    {VK_FORMAT_R64_SINT, {8, 1}},
    {VK_FORMAT_R64_SFLOAT, {8, 1}},
    {VK_FORMAT_R64G64_UINT, {16, 2}},
    {VK_FORMAT_R64G64_SINT, {16, 2}},
    {VK_FORMAT_R64G64_SFLOAT, {16, 2}},
    {VK_FORMAT_R64G64B64_UINT, {24, 3}},
    {VK_FORMAT_R64G64B64_SINT, {24, 3}},
    {VK_FORMAT_R64G64B64_SFLOAT, {24, 3}},
    {VK_FORMAT_R64G64B64A64_UINT, {32, 4}},
    {VK_FORMAT_R64G64B64A64_SINT, {32, 4}},
    {VK_FORMAT_R64G64B64A64_SFLOAT, {32, 4}},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, {4, 3}},
    {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, {4, 3}},
    {VK_FORMAT_D16_UNORM, {2, 1}},
    {VK_FORMAT_X8_D24_UNORM_PACK32, {4, 1}},
    {VK_FORMAT_D32_SFLOAT, {4, 1}},
    {VK_FORMAT_S8_UINT, {1, 1}},
    {VK_FORMAT_D16_UNORM_S8_UINT, {3, 2}},
    {VK_FORMAT_D24_UNORM_S8_UINT, {4, 2}},
    {VK_FORMAT_D32_SFLOAT_S8_UINT, {8, 2}},
};

std::string errorString(VkResult errorCode) {
    switch (errorCode) {
#define STR(r)   \
    case VK_##r: \
        return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
#undef STR
        default:
            return "UNKNOWN_ERROR";
    }
}

std::string physicalDeviceTypeString(VkPhysicalDeviceType type) {
    switch (type) {
#define STR(r)                        \
    case VK_PHYSICAL_DEVICE_TYPE_##r: \
        return #r
        STR(OTHER);
        STR(INTEGRATED_GPU);
        STR(DISCRETE_GPU);
        STR(VIRTUAL_GPU);
#undef STR
        default:
            return "UNKNOWN_DEVICE_TYPE";
    }
}

VkResult createInstance(VkInstance& instance, const char* appName) {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = appName;
    appInfo.pEngineName = appName;
    appInfo.apiVersion = VK_API_VERSION_1_0;
    appInfo.applicationVersion = 1;
    appInfo.engineVersion = 1;

    std::vector<const char*> instanceExtensions = {
        VK_KHR_SURFACE_EXTENSION_NAME};
    // Enable surface extensions depending on os
#if defined(_WIN32)
    instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
    instanceExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    // 和android里的AHardwareBuffer交互

    instanceExtensions.push_back(
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    instanceExtensions.push_back(
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    instanceExtensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
    instanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
    instanceExtensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
    instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
    instanceExtensions.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
    instanceExtensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#endif

#if defined(_WIN32)
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pNext = NULL;
    instInfo.flags = 0;
    instInfo.pApplicationInfo = &appInfo;
    instInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
    instInfo.ppEnabledExtensionNames = instanceExtensions.data();
    // validation
    return vkCreateInstance(&instInfo, nullptr, &instance);
}

VkResult enumerateDevice(VkInstance instance,
                         std::vector<PhysicalDevicePtr>& pDevices) {
    VkResult err;
    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
    assert(gpuCount > 0);
    // Enumerate devices
    pDevices.resize(gpuCount);
    std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
    err =
        vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
    for (uint32_t i = 0; i < gpuCount; i++) {
        pDevices[i] = std::shared_ptr<PhysicalDevice>(new PhysicalDevice());
        pDevices[i]->physicalDevice = physicalDevices[i];
    }
    if (err != VK_SUCCESS) {
        return err;
    }
    for (uint32_t i = 0; i < gpuCount; i++) {
        vkGetPhysicalDeviceProperties(pDevices[i]->physicalDevice,
                                      &pDevices[i]->properties);
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pDevices[i]->physicalDevice,
                                                 &queueFamilyCount, nullptr);
        vkGetPhysicalDeviceMemoryProperties(pDevices[i]->physicalDevice,
                                            &pDevices[i]->mempryProperties);
        if (queueFamilyCount > 0) {
            pDevices[i]->queueFamilyProps.resize(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(
                pDevices[i]->physicalDevice, &queueFamilyCount,
                pDevices[i]->queueFamilyProps.data());
        }
        for (int j = 0; j < pDevices[i]->queueFamilyProps.size(); j++) {
            if ((pDevices[i]->queueFamilyProps[j].queueFlags &
                 VK_QUEUE_GRAPHICS_BIT) == VK_QUEUE_GRAPHICS_BIT) {
                pDevices[i]->queueGraphicsIndexs.push_back(j);
            }
            if ((pDevices[i]->queueFamilyProps[j].queueFlags &
                 VK_QUEUE_COMPUTE_BIT) == VK_QUEUE_COMPUTE_BIT) {
                pDevices[i]->queueComputeIndexs.push_back(j);
            }
        }
    }
    return VK_SUCCESS;
}

int32_t getByteSize(VkFormat format) {
    auto item = formatTable.find(format);
    if (item != formatTable.end()) {
        return item->second.size;
    }
    return 0;
}

bool getMemoryTypeIndex(uint32_t typeBits, VkFlags quirementsMaks,
                        uint32_t& index) {
    const auto& memoryPropertys =
        VulkanManager::Get().physical->mempryProperties;
    for (uint32_t i = 0; i < memoryPropertys.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((memoryPropertys.memoryTypes[i].propertyFlags &
                 quirementsMaks) == quirementsMaks) {
                index = i;
                return true;
            }
        }
        typeBits >>= 1;
    }
    return false;
}

#if defined(__ANDROID__)
// Android shaders are stored as assets in the apk
// So they need to be loaded via the asset manager
VkShaderModule loadShader(AAssetManager* assetManager, const char* fileName,
                          VkDevice device) {
    // Load shader from compressed asset
    AAsset* asset =
        AAssetManager_open(assetManager, fileName, AASSET_MODE_STREAMING);
    assert(asset);
    size_t size = AAsset_getLength(asset);
    assert(size > 0);

    char* shaderCode = new char[size];
    AAsset_read(asset, shaderCode, size);
    AAsset_close(asset);

    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo moduleCreateInfo;
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.pNext = NULL;
    moduleCreateInfo.codeSize = size;
    moduleCreateInfo.pCode = (uint32_t*)shaderCode;
    moduleCreateInfo.flags = 0;

    VK_CHECK_RESULT(
        vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule));

    delete[] shaderCode;

    return shaderModule;
}
#else
VkShaderModule loadShader(const char* fileName, VkDevice device) {
    std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

    if (is.is_open()) {
        size_t size = is.tellg();
        is.seekg(0, std::ios::beg);
        char* shaderCode = new char[size];
        is.read(shaderCode, size);
        is.close();

        assert(size > 0);

        VkShaderModule shaderModule;
        VkShaderModuleCreateInfo moduleCreateInfo{};
        moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCreateInfo.codeSize = size;
        moduleCreateInfo.pCode = (uint32_t*)shaderCode;

        VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, NULL,
                                             &shaderModule));

        delete[] shaderCode;

        return shaderModule;
    } else {
        std::cerr << "Error: Could not open shader file \"" << fileName << "\""
                  << "\n";
        return VK_NULL_HANDLE;
    }
}
#endif

void changeLayout(VkCommandBuffer command, VkImage image,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags oldStageFlags,
                  VkPipelineStageFlags newStageFlags,
                  VkImageAspectFlags aspectMask, VkAccessFlags newAccessFlags) {
    VkImageMemoryBarrier imageMemoryBarrier = {};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.pNext = nullptr;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = 0;
    imageMemoryBarrier.oldLayout = oldLayout;
    imageMemoryBarrier.newLayout = newLayout;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                           1};
    switch (oldLayout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.srcAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            // 必须仅用作传输命令的目标映像,要求启用VK_IMAGE_USAGE_TRANSFER_DST_BIT
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            //该布局旨在用作其内容由主机写入的图像的初始布局，因此无需首先执行布局转换就可以将数据立即写入内存
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;
        default:
            break;
    }
    switch (newLayout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        default:
            break;
    }
    if (newAccessFlags != 0) {
        imageMemoryBarrier.dstAccessMask != newAccessFlags;
    }
    // 等待命令列表里GPU里处理完成
    vkCmdPipelineBarrier(command, oldStageFlags, newStageFlags, 0, 0, nullptr,
                         0, nullptr, 1, &imageMemoryBarrier);
}

}  // namespace vulkan
}  // namespace aoce