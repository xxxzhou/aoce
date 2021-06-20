#include "MfModule.hpp"

#include <mfapi.h>

#include <AoceManager.hpp>
#include <WinHelp.hpp>

#include "videodevice/MFVideoManager.hpp"

namespace aoce {
namespace win {
namespace mf {
    
MfModule::MfModule(/* args */) {}

MfModule::~MfModule() {}

bool MfModule::loadModule() {
    HRESULT hr = CoInitialize(NULL);
    if (SUCCEEDED(hr)) {
        hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        hr = MFStartup(MF_VERSION);
    }
    logHResult(hr, "coInitialize");
    AoceManager::Get().addVideoManager(CameraType::win_mf,
                                       new MFVideoManager());
    return SUCCEEDED(hr);
}

void MfModule::unloadModule() {
    AoceManager::Get().removeVideoManager(CameraType::win_mf);
    HRESULT hr = MFShutdown();
    CoUninitialize();
}

ADD_MODULE(MfModule, aoce_win_mf)

}  // namespace mf
}  // namespace win
}  // namespace aoce
