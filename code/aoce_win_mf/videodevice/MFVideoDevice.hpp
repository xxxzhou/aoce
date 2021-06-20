#pragma once

#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>

#include <videodevice/VideoDevice.hpp>
#include <WinHelp.hpp>
#include <condition_variable>
#include <mutex>

namespace aoce {
namespace win {
namespace mf {

class MFVideoDevice : public VideoDevice, public IMFSourceReaderCallback {
   private:
    long refCount = 0;
    CComPtr<IMFAttributes> pAttributes = nullptr;
    CComPtr<IMFActivate> activate = nullptr;
    CComPtr<IMFMediaSource> source = nullptr;
    CComPtr<IMFSourceReader> sourceReader = nullptr;
    CComPtr<IMFMediaTypeHandler> handle = nullptr;
    //
    std::mutex playMtx;
    //
    std::mutex flushMtx;
    std::condition_variable flushSignal;

    int32_t videoIndex = MF_SOURCE_READER_FIRST_VIDEO_STREAM;
    // 是否异步
    bool bMFAsync = true;

   public:
    MFVideoDevice(/* args */);
    virtual ~MFVideoDevice() override;

   public:
    bool init(IMFActivate* pActivate);
    bool setPlay(bool play);

   public:
    // 摄像机有自己特定输出格式
    virtual void setFormat(int32_t index) override;
    // 打开摄像头
    virtual bool open() override;
    // 关闭摄像头
    virtual bool close() override;

   public:
    // 通过 IMFSourceReaderCallback 继承
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid,
                                                     void** ppvObject) override;
    virtual ULONG STDMETHODCALLTYPE AddRef(void) override;
    virtual ULONG STDMETHODCALLTYPE Release(void) override;
    virtual HRESULT STDMETHODCALLTYPE OnReadSample(HRESULT hrStatus,
                                                   DWORD dwStreamIndex,
                                                   DWORD dwStreamFlags,
                                                   LONGLONG llTimestamp,
                                                   IMFSample* pSample) override;
    virtual HRESULT STDMETHODCALLTYPE OnFlush(DWORD dwStreamIndex) override;
    virtual HRESULT STDMETHODCALLTYPE OnEvent(DWORD dwStreamIndex,
                                              IMFMediaEvent* pEvent) override;
};

}  // namespace mf
}  // namespace win
}  // namespace aoce