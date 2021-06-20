#include "WinRTModule.hpp"

#include <AoceManager.hpp>

#include "RTCaptureWindow.hpp"

namespace aoce {
namespace awinrt {

WinRTModule::WinRTModule(/* args */) {}

WinRTModule::~WinRTModule() {}

bool WinRTModule::loadModule() {
    try {
        /* no contract for IGraphicsCaptureItemInterop, verify 10.0.18362.0 */
        winrt::Windows::Foundation::Metadata::ApiInformation::
            IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 8);
    } catch (...) {
        logMessage(LogLevel::error, "winrt capture not supported");
        return false;
    }
    if (!winrt::Windows::Graphics::Capture::GraphicsCaptureSession::
            IsSupported()) {
        return false;
    }
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    AoceManager::Get().addICaptureWindow(CaptureType::win_rt,
                                         new RTCaptureWindow());
    return true;
}

void WinRTModule::unloadModule() {
    AoceManager::Get().removeICaptureWindow(CaptureType::win_rt);
    winrt::uninit_apartment();
}

ADD_MODULE(WinRTModule, aoce_winrt)

}  // namespace awinrt
}  // namespace aoce