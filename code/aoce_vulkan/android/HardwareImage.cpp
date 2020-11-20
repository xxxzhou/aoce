#include "HardwareImage.hpp"

#include "../vulkan/VulkanManager.hpp"
#include "vulkan_wrapper.h"

// https://source.android.com/devices/graphics/implement-vulkan?hl=zh-cn
// vkQueueSignalReleaseImageANDROID

#define LOAD_PROC(NAME, TYPE) \
    NAME = reinterpret_cast<TYPE>(eglGetProcAddress(#NAME))

using PFNEGLGETNATIVECLIENTBUFFERANDROID =
    EGLClientBuffer(EGLAPIENTRYP)(const AHardwareBuffer *buffer);
using PFNGLEGLIMAGETARGETTEXTURE2DOESPROC = void(GL_APIENTRYP)(GLenum target,
                                                               void *image);
using PFNGLBUFFERSTORAGEEXTERNALEXTPROC = void(GL_APIENTRYP)(GLenum target,
                                                             GLintptr offset,
                                                             GLsizeiptr size,
                                                             void *clientBuffer,
                                                             GLbitfield flags);
using PFNGLMAPBUFFERRANGEPROC = void *(GL_APIENTRYP)(GLenum target,
                                                     GLintptr offset,
                                                     GLsizeiptr length,
                                                     GLbitfield access);
using PFNGLUNMAPBUFFERPROC = void *(GL_APIENTRYP)(GLenum target);
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES;
PFNEGLGETNATIVECLIENTBUFFERANDROID eglGetNativeClientBufferANDROID;
PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR;
PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR;
PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC glFramebufferTextureMultiviewOVR;
PFNGLFRAMEBUFFERTEXTUREMULTISAMPLEMULTIVIEWOVRPROC
glFramebufferTextureMultisampleMultiviewOVR;
PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNGLMAPBUFFERRANGEPROC glMapBufferRange;
PFNGLUNMAPBUFFERPROC glUnmapBuffer;
PFN_vkGetAndroidHardwareBufferPropertiesANDROID
        vkGetAndroidHardwareBufferPropertiesANDROID;
PFN_vkBindImageMemory2KHR vkBindImageMemory2KHR;
namespace aoce {
namespace vulkan {

HardwareImage::HardwareImage(/* args */) {
    vkDevice = VulkanManager::Get().device;
    vkGetAndroidHardwareBufferPropertiesANDROID =
            reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(vkGetDeviceProcAddr(
            vkDevice, "vkGetAndroidHardwareBufferPropertiesANDROID"));
    vkBindImageMemory2KHR = reinterpret_cast<PFN_vkBindImageMemory2KHR>(vkGetDeviceProcAddr(
        vkDevice, "vkBindImageMemory2KHR"));
    // assert(vkGetAndroidHardwareBufferPropertiesANDROID);
    // assert(vkBindImageMemory2KHR);

    LOAD_PROC(glEGLImageTargetTexture2DOES,
              PFNGLEGLIMAGETARGETTEXTURE2DOESPROC);
    // assert(glEGLImageTargetTexture2DOES);
    LOAD_PROC(eglGetNativeClientBufferANDROID,
              PFNEGLGETNATIVECLIENTBUFFERANDROID);
    // assert(eglGetNativeClientBufferANDROID);
    LOAD_PROC(eglCreateImageKHR, PFNEGLCREATEIMAGEKHRPROC);
    // assert(eglCreateImageKHR);
    LOAD_PROC(eglDestroyImageKHR, PFNEGLDESTROYIMAGEKHRPROC);
    // assert(eglDestroyImageKHR);
}

HardwareImage::~HardwareImage() { release(); }

void HardwareImage::release() {
    if (buffer != nullptr) {
#if __ANDROID_API__ >= 26
        AHardwareBuffer_release(buffer);
#endif
        buffer = nullptr;
    }
    if (image) {
        eglDestroyImageKHR(display, image);
        image = nullptr;
    }
    if (vkImage) {
        vkDestroyImage(vkDevice, vkImage, nullptr);
        vkImage = VK_NULL_HANDLE;
    }
    if (memory) {
        vkFreeMemory(vkDevice, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

void HardwareImage::createAndroidBuffer(const ImageFormat &format) {
    // 释放前面释放过的资源
    release();

    this->format = format;
    AHardwareBuffer_Desc usage = {};
    // filling in the usage for HardwareBuffer
    usage.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    usage.height = format.height;
    usage.width = format.width;
    usage.layers = 1;
    usage.rfu0 = 0;
    usage.rfu1 = 0;
    usage.stride = 10;
    usage.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                  AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER |
                  AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
#if __ANDROID_API__ >= 26
    AHardwareBuffer_allocate(&usage, &buffer);
#endif
    bindVK(buffer);
}

// https://android.googlesource.com/platform/cts/+/master/tests/tests/graphics/jni/VulkanTestHelpers.cpp
void HardwareImage::bindVK(AHardwareBuffer *buffer, bool useExternalFormat) {
    AHardwareBuffer_Desc bufferDesc;
#if __ANDROID_API__ >= 26
    AHardwareBuffer_describe(buffer, &bufferDesc);
#else
    bufferDesc.width = format.width;
    bufferDesc.height = format.height;
    bufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    bufferDesc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                       AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER |
                       AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
#endif
    VkAndroidHardwareBufferFormatPropertiesANDROID formatInfo = {
        .sType =
            VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
        .pNext = nullptr,
    };
    VkAndroidHardwareBufferPropertiesANDROID properties = {
        .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
        .pNext = &formatInfo,
    };
    vkGetAndroidHardwareBufferPropertiesANDROID(vkDevice, buffer, &properties);
    VkExternalFormatANDROID externalFormat{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
        .pNext = nullptr,
        .externalFormat = formatInfo.externalFormat,
    };
    VkExternalMemoryImageCreateInfo externalCreateInfo{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = useExternalFormat ? &externalFormat : nullptr,
        .handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
    };
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalCreateInfo, imageInfo.flags = 0u;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format =
        useExternalFormat ? VK_FORMAT_UNDEFINED : formatInfo.format;
    imageInfo.extent = {
        bufferDesc.width,
        bufferDesc.height,
        1u,
    };
    imageInfo.mipLevels = 1u, imageInfo.arrayLayers = 1u;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.queueFamilyIndexCount = 0;
    imageInfo.pQueueFamilyIndices = nullptr;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK_RESULT(vkCreateImage(vkDevice, &imageInfo, nullptr, &vkImage));

    VkImportAndroidHardwareBufferInfoANDROID androidHardwareBufferInfo{
        .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
        .pNext = nullptr,
        .buffer = buffer,
    };
    VkMemoryDedicatedAllocateInfo memoryAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &androidHardwareBufferInfo,
        .image = vkImage,
        .buffer = VK_NULL_HANDLE,
    };
    // android的hardbuffer位置(properties)
    VkMemoryRequirements requires;
    vkGetImageMemoryRequirements(vkDevice, vkImage, &requires);
    uint32_t memoryTypeIndex = 0;
    bool getIndex =
        getMemoryTypeIndex(properties.memoryTypeBits, 0, memoryTypeIndex);
    assert(getIndex);
    VkMemoryAllocateInfo memoryInfo = {};
    memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryInfo.pNext = &memoryAllocateInfo;
    memoryInfo.memoryTypeIndex = memoryTypeIndex;
    memoryInfo.allocationSize = properties.allocationSize;
    VK_CHECK_RESULT(vkAllocateMemory(vkDevice, &memoryInfo, nullptr, &memory));

    VkBindImageMemoryInfo bindImageInfo;
    bindImageInfo.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    bindImageInfo.pNext = nullptr;
    bindImageInfo.image = vkImage;
    bindImageInfo.memory = memory;
    bindImageInfo.memoryOffset = 0;
    VK_CHECK_RESULT(vkBindImageMemory2KHR(vkDevice, 1, &bindImageInfo));

    // android生成一个EGLImageKHR对象,用于gl复制
    EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(buffer);
    assert(native_buffer);
    EGLint attrs[] = {EGL_NONE};
    display = eglGetCurrentDisplay();  // eglGetDisplay(EGL_DEFAULT_DISPLAY);
    image = eglCreateImageKHR(display, EGL_NO_CONTEXT,
                              EGL_NATIVE_BUFFER_ANDROID, native_buffer, attrs);
    assert(image != EGL_NO_IMAGE_KHR);
}

void HardwareImage::bindGL(uint32_t textureId) {
    if(!image){
        return;
    }
    // AHardwareBuffer_lock(AHARDWAREBUFFER_USAGE_CPU_READ_NEVER)
    glBindTexture(GL_TEXTURE_2D,
                  textureId);  // GL_TEXTURE_EXTERNAL_OES GL_TEXTURE_2D
    glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image);
    glBindTexture(GL_TEXTURE_2D, 0);
}

}  // namespace vulkan
}  // namespace aoce