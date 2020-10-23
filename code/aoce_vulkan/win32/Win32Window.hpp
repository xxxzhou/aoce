#pragma once
#include <windows.h>

namespace aoce {
namespace vulkan {
class Win32Window {
   private:
    /* data */
   public:
    Win32Window(/* args */);
    ~Win32Window();

   public:
    HWND hwnd;

   public:
    HWND initWindow(HINSTANCE inst, int width, int height, const char* name,
                    class VulkanWindow* swapChain);
};
}  // namespace common
}  // namespace vulkanx