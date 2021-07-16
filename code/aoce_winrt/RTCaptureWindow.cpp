#include "RTCaptureWindow.hpp"

#include "aoce_win/dx11/Dx11Helper.hpp"

extern "C" {
HRESULT __stdcall CreateDirect3D11DeviceFromDXGIDevice(
    ::IDXGIDevice* dxgiDevice, ::IInspectable** graphicsDevice);

HRESULT __stdcall CreateDirect3D11SurfaceFromDXGISurface(
    ::IDXGISurface* dgxiSurface, ::IInspectable** graphicsSurface);
}

struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1"))
    IDirect3DDxgiInterfaceAccess : ::IUnknown {
    virtual HRESULT __stdcall GetInterface(GUID const& id, void** object) = 0;
};

using namespace aoce::win;

namespace aoce {
namespace awinrt {

RTCaptureWindow::RTCaptureWindow(/* args */) {
    // createDevice11(&device, &ctx);
    CComPtr<IDXGIFactory1> factory = nullptr;
    CComPtr<IDXGIAdapter1> adapter = nullptr;
    IID factoryIID = __uuidof(IDXGIFactory1);
    HRESULT hr;
    hr = CreateDXGIFactory1(factoryIID, (void**)&factory);
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture CreateDXGIFactory1 failed");
    }
    // 后面加上设备选择
    hr = factory->EnumAdapters1(0, &adapter);
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture enumerate DXGIAdapter failed");
    }
    DXGI_ADAPTER_DESC desc = {};
    if (SUCCEEDED(adapter->GetDesc(&desc))) {
        std::wstring wadapterName = desc.Description;
        std::string msg =
            "winrt capture graphics card:" + utf8TString(wadapterName);
        logMessage(LogLevel::info, msg);
    }
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL flOut;
    hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                           D3D11_CREATE_DEVICE_BGRA_SUPPORT, featureLevels,
                           sizeof(featureLevels) / sizeof(D3D_FEATURE_LEVEL),
                           D3D11_SDK_VERSION, &device, &flOut, &ctx);
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture D3D11CreateDevice failed");
    }
    videoFormat.videoType = VideoType::bgra8;
}

RTCaptureWindow::~RTCaptureWindow() {}

bool RTCaptureWindow::startCapture(IWindow* window, bool bSync) {
    if (bCapture) {
        return true;
    }
    hwnd = (HWND)window->getHwnd();
    // get winrt d3ddevice
    CComPtr<IDXGIDevice> dxgiDevice = nullptr;
    HRESULT hr = device->QueryInterface(&dxgiDevice);
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture get IDXGIDevice error");
        return false;
    }
    winrt::com_ptr<IInspectable> inspectable;
    hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, inspectable.put());
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture failed to get WinRT device");
        return false;
    }
    rtDevice = inspectable.as<winrt::IDirect3DDevice>();
    // get winrt GraphicsCaptureItem
    auto activationFactory =
        winrt::get_activation_factory<winrt::GraphicsCaptureItem>();
    auto interopFactory = activationFactory.as<IGraphicsCaptureItemInterop>();
    winrt::GraphicsCaptureItem item{nullptr};
    winrt::guid captureItemGuid =
        winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>();
    hr = interopFactory->CreateForWindow(
        hwnd, captureItemGuid, reinterpret_cast<void**>(winrt::put_abi(item)));
    if (FAILED(hr)) {
        logHResult(hr, "winrt capture failed to CreateForWindow");
        return false;
    }
    // item.Closed({this, &RTCaptureWindow::onClosed});
    framePool = winrt::Direct3D11CaptureFramePool::Create(
        rtDevice, winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2,
        item.Size());
    session = framePool.CreateCaptureSession(item);
    if (bSync) {
        framePool.FrameArrived({this, &RTCaptureWindow::onFrameArrived});
    }
    bCapture = true;
    videoFormat.width = 0;   // item.Size().Width;
    videoFormat.height = 0;  // item.Size().Height;
    videoFormat.fps = 30;
    session.StartCapture();
    return true;
}

void RTCaptureWindow::onClosed(winrt::GraphicsCaptureItem const& sender,
                               winrt::IInspectable const& inspectable) {
    // onObserverEvent(CaptureEventId::close, LogLevel::warn,
    //                 "winrt capture window closed");
    // stopCapture();
}

bool RTCaptureWindow::renderCapture() {
    if (!bCapture) {
        return false;
    }
    if (!validWindow(hwnd)) {
        onObserverEvent(CaptureEventId::lost, LogLevel::warn,
                        "winrt capture window is lost");
        return false;
    }
    winrt::Direct3D11CaptureFrame frame = framePool.TryGetNextFrame();
    if (!frame) {
        return true;
    }
    winrt::SizeInt32 frameSize = frame.ContentSize();
    winrt::com_ptr<ID3D11Texture2D> frameSurface = nullptr;
    auto access = frame.Surface().as<IDirect3DDxgiInterfaceAccess>();
    winrt::check_hresult(access->GetInterface(winrt::guid_of<ID3D11Texture2D>(),
                                              frameSurface.put_void()));
    // D3D11_TEXTURE2D_DESC desc = {};
    // frameSurface->GetDesc(&desc);
    int32_t width = frameSize.Width;
    int32_t height = frameSize.Height;
    if (width == 0 || height == 0) {
        onObserverEvent(CaptureEventId::failed, LogLevel::warn,
                        "winrt capture window rect is zero");
        return false;
    }
    if (width != videoFormat.width || height != videoFormat.height) {
        framePool.Recreate(rtDevice,
                           winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2,
                           frameSize);
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
    ctx->CopyResource(gdiTexture, frameSurface.get());
    if (observer) {
        observer->onCapture(videoFormat, device, gdiTexture);
    }
    return true;
}

void RTCaptureWindow::onFrameArrived(winrt::Direct3D11CaptureFramePool sender,
                                     winrt::IInspectable const& inspectable) {
    renderCapture();
}

void RTCaptureWindow::stopCapture() {
    bCapture = false;
    try {
        if (framePool) {
            framePool.Close();
            framePool = nullptr;
        }
        if (session) {
            session.Close();
            session = nullptr;
        }
    } catch (...) {
        logMessage(LogLevel::info, "stop capture catch ...");
    }
    videoFormat.width = 0;
    videoFormat.height = 0;
}

}  // namespace awinrt
}  // namespace aoce
