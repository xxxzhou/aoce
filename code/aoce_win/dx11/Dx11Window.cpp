#include "Dx11Window.hpp"

namespace aoce {
namespace win {

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    Dx11Window *window =
        reinterpret_cast<Dx11Window *>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
    if (!window) {
        return (DefWindowProc(hWnd, uMsg, wParam, lParam));
    }
    return window->handleMessage(uMsg, wParam, lParam);
}

Dx11Window::Dx11Window(/* args */) {}

Dx11Window::~Dx11Window() {}

LRESULT Dx11Window::handleMessage(UINT msg, WPARAM wparam, LPARAM lparam) {
    PAINTSTRUCT ps;
    HDC hdc;
    switch (msg) {
        case WM_PAINT: {
            hdc = BeginPaint(hwnd, &ps);
            EndPaint(hwnd, &ps);
        } break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wparam, lparam);
            break;
    }
    return 0;
}

void Dx11Window::initDevice() {
    bInitDevice = false;

    HRESULT hr = S_OK;
    RECT rc = {};
    GetClientRect(hwnd, &rc);
    UINT createDeviceFlags = 0;
#if AOCE_DEBUG_TYPE
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] = {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT numDriverTypes = ARRAYSIZE(driverTypes);

    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    UINT numFeatureLevels = ARRAYSIZE(featureLevels);

    DXGI_SWAP_CHAIN_DESC sd = {};
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format =
        DXGI_FORMAT_R8G8B8A8_UNORM;  // DXGI_FORMAT_B8G8R8A8_UNORM,DXGI_FORMAT_R8G8B8A8_UNORM
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes;
         driverTypeIndex++) {
        driverType = driverTypes[driverTypeIndex];
        hr = D3D11CreateDeviceAndSwapChain(
            NULL, driverType, NULL, createDeviceFlags, featureLevels,
            numFeatureLevels, D3D11_SDK_VERSION, &sd, &swapChain, &device,
            &featureLevel, &context);
        if (SUCCEEDED(hr)) {
            break;
        }
    }
    if (FAILED(hr)) {
        logHResult(hr, "create device and swapchain failed");
        return;
    }
    hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID *)&backTex);
    if (FAILED(hr)) {
        logHResult(hr, "get swapchain buffer failed");
        return;
    }
    hr = device->CreateRenderTargetView(backTex, NULL, &renderView);
    if (FAILED(hr)) {
        logHResult(hr, "create render target viewd failed");
        return;
    }
    ID3D11RenderTargetView *renderViews[1] = {renderView};
    context->OMSetRenderTargets(1, renderViews, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width = (FLOAT)width;
    vp.Height = (FLOAT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    context->RSSetViewports(1, &vp);
    bInitDevice = true;
}

void Dx11Window::render() {
    if (!bInitDevice) {
        return;
    }
    if (onTickEvent) {
        onTickEvent(device, backTex);
    }
    swapChain->Present(0, 0);
}

HWND Dx11Window::initWindow(HINSTANCE inst, int width, int height,
                            const char *name, onTickHandle handle) {
    this->width = width;
    this->height = height;
    this->onTickEvent = handle;

    WNDCLASSEX wndClass = {};
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
        logMessage(LogLevel::error, "Could not register window class");
        return nullptr;
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
    SetWindowLongPtr(window, GWLP_USERDATA, (LONG_PTR)this);
    this->hwnd = window;
    // create device11
    initDevice();

    return window;
}

void Dx11Window::run() {
    MSG msg = {0};
    while (WM_QUIT != msg.message) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            render();
        }
    }
}

}  // namespace win
}  // namespace aoce