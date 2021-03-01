#pragma once

// 用于vulkan与dx11交互
// PS:
// https://github.com/krOoze/Hello_Triangle/blob/dxgi_interop/src/WSI/DxgiWsi.h
// https://github.com/roman380/VulkanSdkDemos/blob/d3d11-image-interop/BindImageMemory2/BindImageMemory2.cpp#L154
#include "../vulkan/VulkanTexture.hpp"
#include "aoce_win/DX11/Dx11Helper.hpp"
#include "aoce_win/DX11/Dx11Resource.hpp"

namespace aoce {
namespace vulkan {

class VkWinImage {
   private:
    /* data */
    std::unique_ptr<win::Dx11SharedTex> shardTex = nullptr;
    std::unique_ptr<win::Dx11SharedTex> tempTex = nullptr;
    VkImage vkImage = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ImageFormat format = {};
    VkDevice vkDevice = VK_NULL_HANDLE;
    bool bInit = false;
#if defined(VK_KHR_external_memory_win32)
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR;
    PFN_vkGetMemoryWin32HandlePropertiesKHR vkGetMemoryWin32HandlePropertiesKHR;
#endif  // defined(VK_KHR_external_memory_win32)
   public:
    VkWinImage(/* args */);
    ~VkWinImage();

   public:
    inline bool getInit() { return bInit; }
    inline VkDeviceMemory getDeviceMemory() { return memory; }
    inline const ImageFormat& getFormat() { return format; }
    inline VkImage getImage() { return vkImage; }
    // inline HANDLE getHandle() { return tempTex->sharedHandle; };
    void release();
    void bindDx11(ID3D11Device* device, ImageFormat format);
    void vkCopyTemp(ID3D11Device* device);
    void tempCopyDx11(ID3D11Device* device,ID3D11Texture2D* dx11Tex);
};

}  // namespace vulkan
}  // namespace aoce