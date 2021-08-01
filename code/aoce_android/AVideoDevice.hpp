#pragma once
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <media/NdkImageReader.h>

#include <videodevice/VideoDevice.hpp>
#include <mutex>

namespace aoce {
namespace android {

VideoType getVideoType(int32_t imageFormat);
int32_t getImageFormat(VideoType videoType);

class AVideoDevice : public VideoDevice {
   private:
    /* data */
    ACameraManager* cameraManager = nullptr;
    ACameraMetadata* metadata = nullptr;
    ACameraDevice* ndkDevice = nullptr;
    AImageReader* imageReader = nullptr;
    ACameraOutputTarget* outputTarget = nullptr;
    ACaptureRequest* request = nullptr;
    ACaptureSessionOutputContainer* outputContainer = nullptr;
    ACaptureSessionOutput* sessionOutput = nullptr;
    ACameraCaptureSession* session = nullptr;
    ANativeWindow* surface = nullptr;
    std::mutex procMtx;
    std::condition_variable stopSignal;
    // std::string mid ="";

    friend void imageCallback(void* context, AImageReader* reader);
    friend void onDisconnected(void* context, ACameraDevice* device);
    friend void onError(void* context, ACameraDevice* device, int error);
    friend void onSessionActive(void* context, ACameraCaptureSession* session) ;
    friend void onSessionClosed(void* context, ACameraCaptureSession* session);
    friend void onCaptureFailed(void* context, ACameraCaptureSession* session,
                                     ACaptureRequest* request,
                                     ACameraCaptureFailure* failure);
    friend void onCaptureSequenceAborted(void* context, ACameraCaptureSession* session,
                                              int sequenceId);
public:
    AVideoDevice(/* args */);
    virtual ~AVideoDevice() override;

   public:
    bool init(ACameraManager* manager, const char* id);

    void onClose();

   public:
    // 摄像机有自己特定输出格式
    virtual void setFormat(int32_t index) override;
    // 打开摄像头
    virtual bool open() override;
    // 关闭摄像头
    virtual bool close() override;
};

}  // namespace android
}  // namespace aoce