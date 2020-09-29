#include "MFVideoDevice.hpp"

#include <shlwapi.h>

#include "MediaStruct.hpp"

namespace aoce {
namespace win {
namespace mf {

VideoType getVideoType(const wchar_t* videoName) {
    static std::vector<std::wstring> videoTypeList = {
        L"MFVideoFormat_NV12",  L"MFVideoFormat_YUY2", L"MFVideoFormat_YVYU",
        L"MFVideoFormat_UYVY",  L"MFVideoFormat_MJPG", L"MFVideoFormat_RGB24",
        L"MFVideoFormat_ARGB32"};
    int vindex = -1;
    for (int i = 0; i < videoTypeList.size(); i++) {
        if (wcscmp(videoName, videoTypeList[i].c_str()) == 0) {
            vindex = i;
            break;
        }
    }
    VideoType videoType = (VideoType)(vindex + 1);
    return videoType;
}

MFVideoDevice::MFVideoDevice(/* args */) {
    auto hr = MFCreateAttributes(&pAttributes, 2);
    if (SUCCEEDED(hr)) {
        //开户格式转换，如mgjp 转yuv2
        pAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
        // 异步采集方案
        pAttributes->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, this);
    }
    videoIndex = MF_SOURCE_READER_FIRST_VIDEO_STREAM;
}

MFVideoDevice::~MFVideoDevice() {
    close();
    handle.Release();
    activate.Release();
    pAttributes.Release();
}

bool MFVideoDevice::init(IMFActivate* pActivate) {
    activate = pActivate;
    wchar_t* wpname = nullptr;
    wchar_t* wid = nullptr;
    auto hr = activate->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                           &wpname, nullptr);
    hr = activate->GetAllocatedString(
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &wid, nullptr);
    copywcharstr((wchar_t*)name.data(), wpname, 256);
    copywcharstr((wchar_t*)id.data(), wid, 256);
    CoTaskMemFree(wpname);
    CoTaskMemFree(wid);
    //很多采集设备可以进这步，但是MF读不了，不需要给出错误信息
    hr = activate->ActivateObject(__uuidof(IMFMediaSource), (void**)&source);
    if (!logHResult(hr, "create media source failed.")) {
        return false;
    }
    hr =
        MFCreateSourceReaderFromMediaSource(source, pAttributes, &sourceReader);
    if (!logHResult(hr, "create media source reader failed.")) {
        return false;
    }
    // format清理
    formats.clear();
    // 获取摄像机所有输出格式
    std::vector<MediaType> mediaList;
    if (!getSourceMediaList(source, mediaList, handle)) {
        return false;
    }
    for (int i = 0; i < mediaList.size(); i++) {
        MediaType& mediaType = mediaList[i];
        VideoFormat videoFormat = {};
        videoFormat.index = i;
        videoFormat.width = mediaType.width;
        videoFormat.height = mediaType.height;
        videoFormat.fps = mediaType.frameRate;
        videoFormat.videoType = getVideoType(mediaType.subtypeName);
        if (videoFormat.videoType != VideoType::other)
            formats.push_back(videoFormat);
    }
    if (formats.size() < 1) {
        return false;
    }
    selectFormat = formats[0];
    return true;
}

bool MFVideoDevice::setPlay(bool play) {
    HRESULT hr = S_OK;
    if (play) {
        if (isOpen) {
            return true;
        }
        int i = 0;
        hr = sourceReader->ReadSample(videoIndex, 0, nullptr, nullptr, nullptr,
                                      nullptr);
        //最多试验三次
        while (FAILED(hr) && i++ < 3) {
            hr = sourceReader->ReadSample(videoIndex, 0, nullptr, nullptr,
                                          nullptr, nullptr);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (SUCCEEDED(hr)) {
            isOpen = true;
            logMessage(LogLevel::info, "start reading data.");
            onDeviceAction(VideoHandleId::open, 0);
            return true;
        } else {
            isOpen = false;
            logHResult(hr, "open camera failed.");
            onDeviceAction(VideoHandleId::openFailed, -1);
            return false;
        }
    } else {
        if (!isOpen) {
            return true;
        }
        isOpen = false;
        if (sourceReader) {
            hr = sourceReader->Flush(videoIndex);
            if (logHResult(hr, "fulsh data", LogLevel::warn)) {
                std::unique_lock<std::mutex> lck(flushMtx);
                auto status =
                    flushSignal.wait_for(lck, std::chrono::seconds(3));
                if (status == std::cv_status::timeout) {
                    logMessage(LogLevel::info, "flush data time out");
                }
            }
        }
        onDeviceAction(VideoHandleId::close, 0);
        return true;
    }
}

void MFVideoDevice::setFormat(int32_t index) {
    if (index < 0 || index >= formats.size()) {
        index = 0;
    }
    bool tempOpen = isOpen;
    if (tempOpen && selectIndex != index) {
        close();
    }
    selectIndex = index;
    selectFormat = formats[index];
    if (tempOpen) {
        open();
    }
}

// 打开摄像头,重新设置格式
bool MFVideoDevice::open() {
    std::lock_guard<std::mutex> mtx_locker(playMtx);
    // 检查是否是需要重新打开
    if (!source) {
        HRESULT hr =
            activate->ActivateObject(__uuidof(IMFMediaSource), (void**)&source);
        if (!logHResult(hr, "create media source failed.")) {
            return false;
        }
        hr = MFCreateSourceReaderFromMediaSource(source, pAttributes,
                                                 &sourceReader);
        if (!logHResult(hr, "create media source reader failed.")) {
            return false;
        }
    }
    // 先关闭在读的流，不然设置不了format
    setPlay(false);
    // 重新设置格式
    CComPtr<IMFMediaType> mtype = nullptr;
    CComPtr<IMFMediaType> pType = nullptr;
    auto hr = handle->GetMediaTypeByIndex(selectFormat.index, &mtype);
    if (SUCCEEDED(hr)) {
        // source 应用新的格式
        hr = handle->SetCurrentMediaType(mtype);
        // source reader得到当前的播放格式，如果是压缩的，输出转码
        hr = sourceReader->GetNativeMediaType(videoIndex, selectIndex, &pType);
        if (!logHResult(hr, "the video formt have error media type.")) {
            return false;
        }
        GUID majorType = {0};
        GUID subtype = {0};
        hr = pType->GetGUID(MF_MT_MAJOR_TYPE, &majorType);
        hr = pType->GetGUID(MF_MT_SUBTYPE, &subtype);
        if (majorType != MFMediaType_Video) {
            if (!logHResult(hr, "the video formt is not video.")) {
                return false;
            }
            if (subtype == MFVideoFormat_MJPG) {
                subtype = MFVideoFormat_YUY2;
                hr = pType->SetGUID(MF_MT_SUBTYPE, subtype);
            }
            hr = sourceReader->SetCurrentMediaType(videoIndex, nullptr, pType);
            if (!logHResult(hr, "change video format failed.")) {
                return false;
            }
            // 重新打开数据读取
            setPlay(true);
        }
        return true;
    }
}

// 关闭摄像头
bool MFVideoDevice::close() {
    std::lock_guard<std::mutex> mtx_locker(playMtx);
    // 关闭数据读取,关闭同步flush
    setPlay(false);
    if (source) {
        sourceReader.Release();
        source.Release();
        activate->ShutdownObject();
    }
    return true;
}

HRESULT MFVideoDevice::QueryInterface(REFIID riid, void** ppvObject) {
    static const QITAB qit[] = {
        QITABENT(MFVideoDevice, IMFSourceReaderCallback),
        {0},
    };
    return QISearch(this, qit, riid, ppvObject);
}

ULONG MFVideoDevice::AddRef(void) { return InterlockedIncrement(&refCount); }

ULONG MFVideoDevice::Release(void) {
    ULONG uCount = InterlockedDecrement(&refCount);
    if (uCount == 0) {
        // pAttributes清零时会调用,然后递归清零,逻辑上pAttributes清0时,本身已经delete
        // delete this;
    }
    // For thread safety, return a temporary variable.
    return uCount;
}

HRESULT MFVideoDevice::OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex,
                                    DWORD dwStreamFlags, LONGLONG llTimestamp,
                                    IMFSample* pSample) {
    HRESULT hr = S_OK;
    //人为中断
    if (!isOpen) {
        return hr;
    }
    if (pSample && onVideoFrameEvent) {
        CComPtr<IMFMediaBuffer> pBuffer = nullptr;
        DWORD lenght;
        pSample->GetTotalLength(&lenght);
        hr = pSample->GetBufferByIndex(0, &pBuffer);
        if (pBuffer) {
            unsigned long length = 0;
            unsigned long maxLength = 0;
            pBuffer->GetCurrentLength(&length);
            byte* data = nullptr;
            auto hr = pBuffer->Lock(&data, &length, &length);
            VideoFrame frame = {};
            frame.data[0] = data;
            frame.videoType = selectFormat.videoType;
            frame.width = selectFormat.width;
            frame.height = selectFormat.height;
            frame.timeStamp = getNowTimeStamp();
            onVideoFrameAction(frame);
            pBuffer->Unlock();
        }
    }
    // Request the next frame.
    hr = sourceReader->ReadSample(videoIndex, 0, nullptr, nullptr, nullptr,
                                  nullptr);
    if (FAILED(hr)) {
        isOpen = false;
        onDeviceAction(VideoHandleId::lost, -1);
        logMessage(LogLevel::warn, "Data interruption.");
    }
    return hr;
}

HRESULT MFVideoDevice::OnFlush(DWORD dwStreamIndex) {
    flushSignal.notify_all();
    return S_OK;
}

HRESULT MFVideoDevice::OnEvent(DWORD dwStreamIndex, IMFMediaEvent* pEvent) {
    return S_OK;
}

}  // namespace mf
}  // namespace win
}  // namespace aoce