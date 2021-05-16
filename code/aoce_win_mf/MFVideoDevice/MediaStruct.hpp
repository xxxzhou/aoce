#pragma once

#include <atlbase.h>
#include <d3d11.h>
#include <guiddef.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include <winerror.h>

#include <chrono>
#include <string>
#include <vector>

// https://msdn.microsoft.com/en-us/library/windows/desktop/ff819477(v=vs.85).aspx
// 写文件
struct IMFMediaType;

struct MFMediaType {
    unsigned int frameSize;
    unsigned int height;
    unsigned int width;
    unsigned int yuvMatrix;
    unsigned int videoLighting;
    unsigned int defaultStride;
    unsigned int videoChromaSiting;
    GUID formatType;
    wchar_t* formatTypeName;
    unsigned int fixedSizeSamples;
    unsigned int videoNominalRange;
    unsigned int frameRate;
    unsigned int frameRateLow;
    unsigned int pixelAspectRatio;
    unsigned int pixelAspectRatioLow;
    unsigned int allSamplesIndependent;
    unsigned int frameRateRangeMin;
    unsigned int frameRateRangeMinLow;
    unsigned int sampleSize;
    unsigned int videoPrimaries;
    unsigned int interlaceMode;
    unsigned int frameRateRangeMax;
    unsigned int frameRateRangeMaxLow;
    unsigned int bitRate;
    GUID majorType;
    wchar_t* majorTypeName;
    GUID subtype;
    wchar_t* subtypeName;
};

bool getSourceMediaList(IMFMediaSource* source,
                        std::vector<MFMediaType>& mediaTypeList,
                        CComPtr<IMFMediaTypeHandler>& handle);
