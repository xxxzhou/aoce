#include "MCaptureWindow.hpp"

namespace aoce {
namespace win {

MCaptureWindow::MCaptureWindow(/* args */) {
    createDevice11(&device, &ctx);
    videoFormat.videoType = VideoType::bgra8;
}

MCaptureWindow::~MCaptureWindow() {}

bool MCaptureWindow::startCapture(IWindow* window, bool bSync) {
    if (bCapture) {
        return true;
    }
    hwnd = (HWND)window->getHwnd();
    this->bSync = bSync;

    bCapture = true;
    videoFormat.width = 0;
    videoFormat.height = 0;
    videoFormat.fps = 30;
    if (bSync) {
        std::thread runThread([&]() {
            while (bCapture) {
                renderCapture();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            stopSignal.notify_all();
        });
        runThread.detach();
    }
    return true;
}

bool MCaptureWindow::renderCapture() {
    if (!bCapture) {
        return false;
    }
    if (!validWindow(hwnd)) {
        onObserverEvent(CaptureEventId::lost, LogLevel::warn, "window is lost");
    }
    RECT rect = {};
    ::GetWindowRect(hwnd, &rect);
    int32_t width = rect.right - rect.left;
    int32_t height = rect.bottom - rect.top;
    if (width == 0 || height == 0) {
        onObserverEvent(CaptureEventId::failed, LogLevel::warn,
                        "window rect is zero");
        return false;
    }
    if (width != videoFormat.width || height != videoFormat.height) {
        // 释放原来纹理
        gdiTexture.Release();
        videoFormat.width = width;
        videoFormat.height = height;
        bool bCreate =
            createGUITextureBuffer(device, width, height, &gdiTexture);
        if (!bCreate) {
            onObserverEvent(CaptureEventId::failed, LogLevel::warn,
                            "create dx11 gui texture failed");
        }
        if (observer) {
            observer->onResize(width, height);
        }
    }
    // 得到窗口DC,如果中间返回自动释放
    std::unique_ptr<HDC__, std::function<void(HDC)>> uhwnd(
        ::GetWindowDC(hwnd), [=](HDC x) { ReleaseDC(hwnd, x); });
    CComPtr<IDXGISurface1> surface = nullptr;
    HRESULT hr =
        gdiTexture->QueryInterface(_uuidof(IDXGISurface1), (LPVOID*)&surface);
    if (FAILED(hr)) {
        // logHResult(hr, "failed get hdc or get DXGI surface");
        onObserverEvent(CaptureEventId::failed, LogLevel::warn,
                        "failed get hdc or get DXGI surface");
        return false;
    }
    HDC gdiHdc = nullptr;
    // 得到渲染纹理DC
    hr = surface->GetDC(FALSE, &gdiHdc);
    if (FAILED(hr)) {
        // logHResult(hr, "failed get hdc or get DXGI surface");
        surface->ReleaseDC(NULL);
        onObserverEvent(CaptureEventId::failed, LogLevel::warn,
                        "failed get surface dc");
        return false;
    }
    std::unique_ptr<HDC__, std::function<void(HDC)>> ugdiHdc(
        gdiHdc, [=](HDC) { surface->ReleaseDC(NULL); });
    auto targetCaps = GetDeviceCaps(uhwnd.get(), RASTERCAPS);
    auto gdiCaps = GetDeviceCaps(ugdiHdc.get(), RASTERCAPS);
    if (!(targetCaps & RC_BITBLT) || !(gdiCaps & RC_BITBLT)) {
        onObserverEvent(
            CaptureEventId::failed, LogLevel::warn,
            "capturing window device context does not support BitBlt");
        return false;
    }
    auto err =
        BitBlt(ugdiHdc.get(), 0, 0, width, height, uhwnd.get(), 0, 0, SRCCOPY);
    // 一定要有,否则数据抓不到
    ugdiHdc.reset();
    if (err == FALSE) {
        onObserverEvent(CaptureEventId::failed, LogLevel::warn,
                        "failed to BitBlt");
        return false;
    }
    if (observer) {
        observer->onCapture(videoFormat, device, gdiTexture);
    }
    return true;
}

void MCaptureWindow::stopCapture() {
    if (!bCapture) {
        return;
    }
    bCapture = false;
    if (bSync) {
        std::unique_lock<std::mutex> lck(stopMtx);
        // 等待preSignal信号回传
        auto status = stopSignal.wait_for(lck, std::chrono::seconds(2));
        if (status == std::cv_status::timeout) {
            logMessage(LogLevel::warn, "capturing window close time out.");
        }
    }
    videoFormat.width = 0;
    videoFormat.height = 0;
}

}  // namespace win
}  // namespace aoce