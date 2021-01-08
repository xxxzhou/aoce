#pragma once

#include "Dx11Helper.hpp"

namespace aoce {
namespace win {

class ACOE_WIN_EXPORT Dx11Window {
   private:
    /* data */
    D3D_DRIVER_TYPE driverType = D3D_DRIVER_TYPE_NULL;
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    HWND hwnd = nullptr;
    int width = 0;
    int height = 0;
    CComPtr<ID3D11Device> device = nullptr;
    CComPtr<ID3D11DeviceContext> context = nullptr;
    CComPtr<IDXGISwapChain> swapChain = nullptr;
    CComPtr<ID3D11RenderTargetView> renderView = nullptr;
    CComPtr<ID3D11Texture2D> backTex = nullptr;

    bool bInitDevice = false;
    onTickHandle onTickEvent = nullptr;

   private:
    friend LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam,
                                    LPARAM lParam);

   public:
    Dx11Window(/* args */);
    ~Dx11Window();

   private:
    LRESULT handleMessage(UINT msg, WPARAM wparam, LPARAM lparam);
    void initDevice();
    void render();

   public:
    HWND initWindow(HINSTANCE inst, int width, int height, const char* name,
                    onTickHandle handle = nullptr);

    void run();
};

}  // namespace win
}  // namespace aoce