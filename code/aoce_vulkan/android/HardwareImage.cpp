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

static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES =
    nullptr;
static PFNEGLGETNATIVECLIENTBUFFERANDROID eglGetNativeClientBufferANDROID =
    nullptr;
static PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR = nullptr;
static PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR = nullptr;
static PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC
    glFramebufferTextureMultiviewOVR = nullptr;
static PFNGLFRAMEBUFFERTEXTUREMULTISAMPLEMULTIVIEWOVRPROC
    glFramebufferTextureMultisampleMultiviewOVR = nullptr;
static PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT = nullptr;
static PFNGLMAPBUFFERRANGEPROC glMapBufferRange = nullptr;
static PFNGLUNMAPBUFFERPROC glUnmapBuffer = nullptr;
static PFN_vkGetAndroidHardwareBufferPropertiesANDROID
    vkGetAndroidHardwareBufferPropertiesANDROID = nullptr;
static PFN_vkBindImageMemory2KHR vkBindImageMemory2KHR = nullptr;

namespace aoce {
namespace vulkan {

bool supportHardwareImage(VkDevice device) {
    vkGetAndroidHardwareBufferPropertiesANDROID =
        reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
            vkGetDeviceProcAddr(device,
                                "vkGetAndroidHardwareBufferPropertiesANDROID"));
    if (!vkGetAndroidHardwareBufferPropertiesANDROID) {
        logMessage(LogLevel::warn,
                   "vkGetAndroidHardwareBufferPropertiesANDROID is null");
        return false;
    }
    vkBindImageMemory2KHR = reinterpret_cast<PFN_vkBindImageMemory2KHR>(
        vkGetDeviceProcAddr(device, "vkBindImageMemory2KHR"));
    if (!vkBindImageMemory2KHR) {
        logMessage(LogLevel::warn, "vkBindImageMemory2KHR is null");
        return false;
    }
    if (!eglGetProcAddress) {
        logMessage(LogLevel::warn, "eglGetProcAddress is null");
        return false;
    }
    LOAD_PROC(glEGLImageTargetTexture2DOES,
              PFNGLEGLIMAGETARGETTEXTURE2DOESPROC);
    if (!glEGLImageTargetTexture2DOES) {
        logMessage(LogLevel::warn, "glEGLImageTargetTexture2DOES is null");
        return false;
    }
    LOAD_PROC(eglGetNativeClientBufferANDROID,
              PFNEGLGETNATIVECLIENTBUFFERANDROID);
    if (!eglGetNativeClientBufferANDROID) {
        logMessage(LogLevel::warn, "eglGetNativeClientBufferANDROID is null");
        return false;
    }
    LOAD_PROC(eglCreateImageKHR, PFNEGLCREATEIMAGEKHRPROC);
    if (!eglCreateImageKHR) {
        logMessage(LogLevel::warn, "eglCreateImageKHR is null");
        return false;
    }
    LOAD_PROC(eglDestroyImageKHR, PFNEGLDESTROYIMAGEKHRPROC);
    if (!eglDestroyImageKHR) {
        logMessage(LogLevel::warn, "eglDestroyImageKHR is null");
        return false;
    }
    return true;
}

HardwareImage::HardwareImage(/* args */) {
    vkDevice = VulkanManager::Get().device;
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
    if(surface != EGL_NO_SURFACE){
        eglDestroySurface(display,surface);
        surface = EGL_NO_SURFACE;
    }
    if(context != EGL_NO_CONTEXT){
        eglDestroyContext(display,context);
        context = EGL_NO_CONTEXT;
    }
    if(display){
        eglTerminate(display);
        display = EGL_NO_DISPLAY;
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
    bindVK(buffer);
#endif
    textureId = -1;
}

// https://android.googlesource.com/platform/cts/+/master/tests/tests/graphics/jni/VulkanTestHelpers.cpp
void HardwareImage::bindVK(AHardwareBuffer *buffer, bool useExternalFormat) {
    AHardwareBuffer_Desc bufferDesc = {};
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
    imageInfo.pNext = &externalCreateInfo;
    imageInfo.flags = 0u;
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

    // android绑定AHardwareBuffer与egl image
    EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(buffer);
    if(!native_buffer){
        logMessage(LogLevel::error, "eglGetNativeClientBufferANDROID failed");
        return;
    }
    if(!initContext()){
        logMessage(LogLevel::error, "hardwareImage initContext failed");
        return;
    }
    saveContext();
    makeCurrent();
    EGLint attrs[] = {EGL_NONE};
    image = eglCreateImageKHR(display, context, EGL_NATIVE_BUFFER_ANDROID,
                              native_buffer, attrs);
    // assert(image != EGL_NO_IMAGE_KHR);
    if (image == EGL_NO_IMAGE_KHR) {
        int32_t errorId = eglGetError();
        logMessage(LogLevel::error,
                   "not create hardware image,error id" + errorId);
    } else {
        logMessage(LogLevel::info, "hardware image create success.");
    }
    restoreContext();
}

void HardwareImage::bindGL(uint32_t textureId, uint32_t texType) {
    if (!image) {
        return;
    }
    int bindType = GL_TEXTURE_2D;
    if (texType > 0) {
        bindType = texType;
    }
    saveContext();
    makeCurrent();

    this->textureId = textureId;
    // AHardwareBuffer_lock(AHARDWAREBUFFER_USAGE_CPU_READ_NEVER)
    // glActiveTexture(GL_TEXTURE0);
    glBindTexture(bindType,
                  textureId);  // GL_TEXTURE_EXTERNAL_OES GL_TEXTURE_2D
    glEGLImageTargetTexture2DOES(bindType, image);
    // glTexParameteri(bindType, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(bindType, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(bindType, 0);

    restoreContext();
}

bool HardwareImage::initContext() {
    EGLContext shareContext = EGL_NO_CONTEXT;
    // eglGetCurrentDisplay()/eglGetDisplay(EGL_DEFAULT_DISPLAY)
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        logMessage(LogLevel::error, "unable to get egl14 display");
        return false;
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (!eglInitialize(display, &major, &minor)) {
        return false;
    }
    EGLConfig config = nullptr;
    EGLint numConfigs = 0;
    const EGLint configSpec[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                                 EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
    if (!eglChooseConfig(display, configSpec, &config, 1, &numConfigs)) {
        return false;
    }
    const EGLint contextAttribsES2[] = {EGL_CONTEXT_CLIENT_VERSION, 2,
                                        EGL_NONE};
    const EGLint contextAttribsES31[] = {EGL_CONTEXT_MAJOR_VERSION_KHR, 3,
                                           EGL_CONTEXT_MINOR_VERSION_KHR, 1,
                                           EGL_NONE};
    context = eglCreateContext(display,config,shareContext,major ==3?contextAttribsES31:contextAttribsES2);
    surface = EGL_NO_SURFACE;
    return true;
}

void HardwareImage::saveContext() {
    oldDisplay = eglGetCurrentDisplay();
    oldContext = eglGetCurrentContext();
    oldSurfaceDraw = eglGetCurrentSurface(EGL_DRAW);
    oldSurfaceRead = eglGetCurrentSurface(EGL_READ);
}

void HardwareImage::makeCurrent() {
    eglMakeCurrent(display,surface,surface,context);
}

void HardwareImage::restoreContext() {
    if(!oldDisplay || oldDisplay == display){
        return;
    }
    eglMakeCurrent(oldDisplay,oldSurfaceDraw,oldSurfaceRead,oldContext);
}

}  // namespace vulkan
}  // namespace aoce