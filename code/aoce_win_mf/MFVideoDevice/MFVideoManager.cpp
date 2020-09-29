#include "MFVideoManager.hpp"

#include "MFVideoDevice.hpp"
namespace aoce {
namespace win {
namespace mf {

MFVideoManager::MFVideoManager(/* args */) {}

MFVideoManager::~MFVideoManager() { videoList.clear(); }

void MFVideoManager::getDevices() {
    // shared 引用清零自动delete
    videoList.clear();
    CComPtr<IMFAttributes> pAttributes = nullptr;
    auto hr = MFCreateAttributes(&pAttributes, 1);
    if (SUCCEEDED(hr)) {
        hr = pAttributes->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        IMFActivate** ppDevices = nullptr;
        UINT32 count = -1;
        hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
        if (SUCCEEDED(hr) && count > 0) {
            for (auto i = 0; i < count; i++) {
                CComPtr<IMFMediaSource> msource = nullptr;
                hr = ppDevices[i]->ActivateObject(__uuidof(IMFMediaSource),
                                                  (void**)&msource);
                if (FAILED(hr)) {
                    logHResult(hr, "create media soure failed.");
                } else {
                    auto videoPtr = std::make_shared<MFVideoDevice>();
                    if (videoPtr->init(ppDevices[i])) {
                        this->videoList.push_back(std::move(videoPtr));
                    }
                }
                ppDevices[i]->Release();
            }
        }
        CoTaskMemFree(ppDevices);
    }
}

}  // namespace mf
}  // namespace win
}  // namespace aoce