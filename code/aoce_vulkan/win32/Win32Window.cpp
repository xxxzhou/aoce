#include "Win32Window.hpp"
#include <iostream>
#include "../vulkan/VulkanWindow.hpp"

namespace aoce {
namespace vulkan {

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    VulkanWindow *swapChain =
        reinterpret_cast<VulkanWindow *>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
    if (!swapChain) {
        return (DefWindowProc(hWnd, uMsg, wParam, lParam));
    }
    return swapChain->handleMessage(uMsg, wParam, lParam);
}

Win32Window::Win32Window(/* args */) {}

Win32Window::~Win32Window() {}

HWND Win32Window::initWindow(HINSTANCE inst, int width, int height,
                             const char *name, class VulkanWindow *swapChain) {
    WNDCLASSEX wndClass;

    wndClass.cbSize = sizeof(WNDCLASSEX);
    wndClass.style = CS_HREDRAW | CS_VREDRAW;
    wndClass.lpfnWndProc = WndProc;
    wndClass.cbClsExtra = 0;
    wndClass.cbWndExtra = 0;
    wndClass.hInstance = inst;
    wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndClass.lpszMenuName = NULL;
    wndClass.lpszClassName = name;
    wndClass.hIconSm = LoadIcon(NULL, IDI_WINLOGO);
    if (!RegisterClassEx(&wndClass)) {
        std::cout << "Could not register window class!\n";
        fflush(stdout);
        exit(1);
    }
    RECT wr = {0, 0, width, height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window =
        CreateWindowEx(WS_EX_APPWINDOW,
                       name,                  // class name
                       name,                  // app name
                       WS_OVERLAPPEDWINDOW |  // window style
                           WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                       100, 100,            // x/y coords
                       wr.right - wr.left,  // width
                       wr.bottom - wr.top,  // height
                       NULL,                // handle to parent
                       NULL,                // handle to menu
                       inst,                // hInstance
                       NULL);               // no extra parameters
    SetForegroundWindow(window);
    SetWindowLongPtr(window, GWLP_USERDATA, (LONG_PTR)swapChain);
    this->hwnd = window;
    return window;
}

}  // namespace vulkan
}  // namespace aoce