#pragma once
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <vulkan/vulkan.h>

#include <Aoce.hpp>

#include "../vulkan/VulkanTexture.hpp"

namespace aoce {
namespace vulkan {

// https://developer.android.com/ndk/reference/group/a-hardware-buffer
// AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM VK_FORMAT_R8G8B8A8_UNORM GL_RGBA8
// AHARDWAREBUFFER_FORMAT_S8_UINT VK_FORMAT_S8_UINT GL_STENCIL_INDEX8

bool supportHardwareImage(VkDevice device);

class HardwareImage {
   private:
    /* data */
    AHardwareBuffer *buffer = nullptr;
    VkDevice vkDevice = VK_NULL_HANDLE;
    VkImage vkImage = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    EGLImageKHR image = nullptr;

    EGLDisplay display = nullptr;
    EGLContext context = nullptr;
    EGLSurface surface = nullptr;

    EGLDisplay oldDisplay = nullptr;
    EGLContext oldContext = nullptr;
    EGLSurface oldSurfaceDraw = nullptr;
    EGLSurface oldSurfaceRead = nullptr;

    ImageFormat format = {};
    int32_t textureId = -1;

    bool bInit = false;

   public:
    HardwareImage(/* args */);
    ~HardwareImage();

    void release();

   public:
    inline VkImage getImage() { return vkImage; }
    inline AHardwareBuffer *getHarderBuffer() { return buffer; }
    inline const ImageFormat &getFormat() { return format; }
    inline const int32_t getTextureId() { return textureId; }

   private:
    // https://android.googlesource.com/platform/cts/+/master/tests/tests/graphics/jni/VulkanTestHelpers.cpp
    void bindVK(AHardwareBuffer *buffer, bool useExternalFormat = false);

   public:
    // gpu输出资源创建
    void createAndroidBuffer(const ImageFormat &format);
    void bindGL(uint32_t textureId, uint32_t texType = 0);

   private:
    // UE4上GPU交互更改成独立Context模式
    bool initContext();
    void saveContext();
    void makeCurrent();
    void restoreContext();
};

}  // namespace vulkan
}  // namespace aoce