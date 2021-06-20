#include "MediaStruct.hpp"

#include <Mfapi.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <string>

#include "FormatReader.hpp"

bool getSourceMediaList(IMFMediaSource* source,
                        std::vector<MFMediaType>& mediaTypeList,
                        CComPtr<IMFMediaTypeHandler>& handle) {
    CComPtr<IMFPresentationDescriptor> pd = nullptr;
    CComPtr<IMFStreamDescriptor> sd = nullptr;
    // CComPtr<IMFMediaTypeHandler> handle = nullptr;

    BOOL bSelected = false;
    unsigned long types = 0;
    auto hr = source->CreatePresentationDescriptor(&pd);
    if (FAILED(hr)) {
        return false;
    }
    hr = pd->GetStreamDescriptorByIndex(0, &bSelected, &sd);
    if (FAILED(hr)) {
        return false;
    }
    hr = sd->GetMediaTypeHandler(&handle);
    if (FAILED(hr)) {
        return false;
    }
    hr = handle->GetMediaTypeCount(&types);
    if (FAILED(hr)) {
        return false;
    }
    for (int i = 0; i < types; i++) {
        CComPtr<IMFMediaType> type = nullptr;
        hr = handle->GetMediaTypeByIndex(i, &type);
        if (FAILED(hr)) {
            return false;
        }
        MFMediaType mediaType = {};
        if (FormatReader::Read(type, mediaType)) {
            mediaTypeList.push_back(mediaType);
        }
    }
    return true;
}
