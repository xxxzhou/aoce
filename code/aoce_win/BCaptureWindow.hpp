#pragma once

#include "WinHelp.hpp"
#include "dx11/Dx11Helper.hpp"

namespace aoce {
namespace win {

class ACOE_WIN_EXPORT BCaptureWindow : public ICaptureWindow {
   private:
    /* data */
   protected:
    ICaptureObserver* observer = nullptr;
    bool bCapture = false;
    CComPtr<ID3D11Device> device = nullptr;
    CComPtr<ID3D11DeviceContext> ctx = nullptr;
    CComPtr<ID3D11Texture2D> gdiTexture = nullptr;
    HWND hwnd = nullptr;
    VideoFormat videoFormat = {};

   protected:
    void onObserverEvent(CaptureEventId eventId, LogLevel level,
                         const char* msg);

   public:
    virtual void setObserver(ICaptureObserver* observer) override;
    virtual bool bCapturing() override;

   public:
    BCaptureWindow(/* args */);
    virtual ~BCaptureWindow();
};

}  // namespace win
}  // namespace aoce