#include "AVideoDevice.hpp"

#include <media/NdkImage.h>

#include <thread>

namespace aoce {
namespace android {

VideoType getVideoType(int32_t imageFormat) {
    switch (imageFormat) {
        case AIMAGE_FORMAT_RGBA_8888:
        case AIMAGE_FORMAT_RGBX_8888:
            return VideoType::rgba8;
        case AIMAGE_FORMAT_RGB_888:
            return VideoType::rgb8;
        case AIMAGE_FORMAT_YUV_420_888:
            // 这里不好确定,很有可能是NV12(需要通过AImageReader_ImageListener/imageCallback来确定)
            return VideoType::yuv420P;
            //        case AIMAGE_FORMAT_JPEG:
            //            return VideoType::mjpg;
        case AIMAGE_FORMAT_DEPTH16:
            return VideoType::depth16u;
    }
    return VideoType::other;
}

int32_t getImageFormat(VideoType videoType) {
    switch (videoType) {
        case VideoType::rgba8:
            return AIMAGE_FORMAT_RGBA_8888;
        case VideoType::rgb8:
            return AIMAGE_FORMAT_RGB_888;
        case VideoType::yuv420P:
            return AIMAGE_FORMAT_YUV_420_888;
        case VideoType::mjpg:
            return AIMAGE_FORMAT_JPEG;
        case VideoType::depth16u:
            return AIMAGE_FORMAT_DEPTH16;
    }
    return 0;
}

// ACameraDevice_stateCallbacks
void onDisconnected(void* context, ACameraDevice* device) {
    AVideoDevice* adevice = (AVideoDevice*)context;
    adevice->onDeviceAction(VideoHandleId::lost, -1);
    adevice->isOpen = false;
}

void onError(void* context, ACameraDevice* device, int error) {
    AVideoDevice* adevice = (AVideoDevice*)context;
    adevice->onDeviceAction(VideoHandleId::unKnownError, error);
}

// ACameraCaptureSession_stateCallbacks
void onSessionActive(void* context, ACameraCaptureSession* session) {
    AVideoDevice* adevice = (AVideoDevice*)context;
    adevice->isOpen = true;
}

void onSessionReady(void* context, ACameraCaptureSession* session) {}

void onSessionClosed(void* context, ACameraCaptureSession* session) {
    AVideoDevice* adevice = (AVideoDevice*)context;
    adevice->isOpen = false;
}

// ACameraCaptureSession_captureCallbacks
void onCaptureFailed(void* context, ACameraCaptureSession* session,
                     ACaptureRequest* request, ACameraCaptureFailure* failure) {
    AVideoDevice* adevice = (AVideoDevice*)context;
    adevice->onDeviceAction(VideoHandleId::openFailed, 0);
}

void onCaptureSequenceCompleted(void* context, ACameraCaptureSession* session,
                                int sequenceId, int64_t frameNumber) {}

void onCaptureSequenceAborted(void* context, ACameraCaptureSession* session,
                              int sequenceId) {
    AVideoDevice* adevice = (AVideoDevice*)context;
    adevice->onDeviceAction(VideoHandleId::lost, sequenceId);
}

void onCaptureCompleted(void* context, ACameraCaptureSession* session,
                        ACaptureRequest* request,
                        const ACameraMetadata* result) {}

// static std::mutex procMtx;

// AImageReader_ImageListener
void imageCallback(void* context, AImageReader* reader) {
    AVideoDevice* device = (AVideoDevice*)context;
    if (device == nullptr) {
        return;
    }
    // std::lock_guard<std::mutex> mtx_locker(device->procMtx);
    AImage* image = nullptr;
    if (AImageReader_acquireLatestImage(reader, &image) != AMEDIA_OK) {
        return;
    }
    // Try to process data without blocking the callback
    uint8_t* data = nullptr;
    int32_t numPlanes = 0;
    media_status_t mstatus = AImage_getNumberOfPlanes(image, &numPlanes);
    if (mstatus == AMEDIA_OK) {
        int64_t timestamp = 0;
        AImage_getTimestamp(image, &timestamp);
        VideoFrame frame = {};
        for (int32_t i = 0; i < numPlanes; i++) {
            int32_t len = 0;
            int32_t rowStride = 0;
            AImage_getPlaneData(image, i, &data, &len);
            AImage_getPlaneRowStride(image, i, &rowStride);
            frame.data[i] = data;
            frame.dataAlign[i] = rowStride;
        }
        frame.videoType = device->selectFormat.videoType;
        if (numPlanes == 3 && frame.dataAlign[0] == frame.dataAlign[1]) {
            frame.videoType = VideoType::nv12;
        }
        frame.width = device->selectFormat.width;
        frame.height = device->selectFormat.height;
        frame.timeStamp = timestamp;
        device->onVideoFrameAction(frame);
        // AImage_delete(image);
    }
    AImage_delete(image);
    //    if(!device->isOpen){
    //        device->onClose();
    //    }
    //    device->stopSignal.notify_all();
}

AVideoDevice::AVideoDevice(/* args */) {}

AVideoDevice::~AVideoDevice() {
    close();
    if (metadata) {
        ACameraMetadata_free(metadata);
        metadata = nullptr;
    }
}

void AVideoDevice::setFormat(int32_t index) {
    selectIndex = index;
    selectFormat = formats[index];
}

bool AVideoDevice::open() {
    ACameraDevice_stateCallbacks cameraDeviceCallbacks = {
        .context = this,
        .onDisconnected = onDisconnected,
        .onError = onError,
    };
    camera_status_t status = ACameraManager_openCamera(
        cameraManager, id.c_str(), &cameraDeviceCallbacks, &ndkDevice);
    if (status == ACAMERA_OK) {
        status = ACameraDevice_createCaptureRequest(ndkDevice, TEMPLATE_PREVIEW,
                                                    &request);
        if (ACaptureRequest_setEntry_i32(request, ACAMERA_CONTROL_AE_TARGET_FPS_RANGE,
                                         2, framerateRange) != ACAMERA_OK) {
            logMessage(LogLevel::warn,"failed to set target fps range in capture request");
        }
        media_status_t mstatus = AImageReader_new(
            selectFormat.width, selectFormat.height,
            getImageFormat(selectFormat.videoType), 4, &imageReader);
        if (mstatus == AMEDIA_OK) {
            if (AImageReader_getWindow(imageReader, &surface) == AMEDIA_OK) {
                AImageReader_ImageListener listener{
                    .context = this,
                    .onImageAvailable = imageCallback,
                };
                AImageReader_setImageListener(imageReader, &listener);
                if (mstatus == AMEDIA_OK) {
                    ANativeWindow_acquire(surface);
                    ACameraOutputTarget_create(surface, &outputTarget);
                    ACaptureRequest_addTarget(request, outputTarget);
                    ACaptureSessionOutputContainer_create(&outputContainer);
                    ACaptureSessionOutput_create(surface, &sessionOutput);
                    ACaptureSessionOutputContainer_add(outputContainer,
                                                       sessionOutput);
                    ACameraCaptureSession_stateCallbacks sessionStateCallbacks{
                        .context = this,
                        .onActive = onSessionActive,
                        .onReady = onSessionReady,
                        .onClosed = onSessionClosed};
                    ACameraDevice_createCaptureSession(
                        ndkDevice, outputContainer, &sessionStateCallbacks,
                        &session);
                    ACameraCaptureSession_captureCallbacks captureCallbacks{
                        .context = nullptr,
                        .onCaptureStarted = nullptr,
                        .onCaptureProgressed = nullptr,
                        .onCaptureCompleted = onCaptureCompleted,
                        .onCaptureFailed = onCaptureFailed,
                        .onCaptureSequenceCompleted =
                            onCaptureSequenceCompleted,
                        .onCaptureSequenceAborted = onCaptureSequenceAborted,
                        .onCaptureBufferLost = nullptr,
                    };
                    status = ACameraCaptureSession_setRepeatingRequest(
                        session, &captureCallbacks, 1, &request, nullptr);
                    return status == ACAMERA_OK;
                }
            }
        }
    }
    return false;
}

bool AVideoDevice::close() {
    // std::lock_guard<std::mutex> mtx_locker(procMtx);
    if (!isOpen) {
        return true;
    }
    onClose();
    return true;
    //    std::unique_lock<std::mutex> lck(procMtx);
    //    // 等待stopSignal信号回传
    //    auto status = stopSignal.wait_for(lck, std::chrono::seconds(2));
    //    if (status == std::cv_status::timeout) {
    //        logMessage(LogLevel::warn, "android ndk camera close time out.");
    //        return false;
    //    }
    //    return true;
}

bool AVideoDevice::init(ACameraManager* manager, const char* id) {
    cameraManager = manager;
    this->id = id;
    this->name = id;
    // copycharstr(this->id.data(), id, AOCE_VIDEO_MAX_NAME);

    ACameraManager_getCameraCharacteristics(cameraManager, id, &metadata);
    ACameraMetadata_const_entry entry = {};
    ACameraMetadata_getConstEntry(
        metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry);
    for (int i = 0; i < entry.count; i += 4) {
        int32_t input = entry.data.i32[i + 3];
        int32_t format = entry.data.i32[i + 0];
        // We are only interested in output streams, so skip input stream
        if (input) {
            continue;
        }
        VideoFormat videoFormat = {};
        videoFormat.width = entry.data.i32[i + 1];
        videoFormat.height = entry.data.i32[i + 2];
        videoFormat.videoType = getVideoType(format);
        videoFormat.index = formats.size();
        if (videoFormat.videoType != VideoType::other) {
            formats.push_back(videoFormat);
        }
    }
    if (formats.size() < 1) {
        return false;
    }
    // FPS
    ACameraMetadata_const_entry frameRates = {};
    if (ACameraMetadata_getConstEntry(metadata,
                                      ACAMERA_CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES,
                                      &frameRates) == ACAMERA_OK) {
        int currentBestMatch = -1;
        bool bFindFps = false;
        for (uint32_t i = 0; i < frameRates.count; i += 2) {
            auto fpsMin = frameRates.data.i32[i + 0];
            auto fpsMax = frameRates.data.i32[i + 1];
            if(fpsMax == fps){
                if(fpsMax == fpsMin){
                    bFindFps = true;
                    framerateRange[0] = fpsMin;
                    framerateRange[0] = fpsMax;
                    break;
                }else if (currentBestMatch >= 0) {
                    int32_t oldCurrentMin = frameRates.data.i32[currentBestMatch * 2 + 0];
                    if (fpsMin > oldCurrentMin) {
                        currentBestMatch = i;
                    }
                } else {
                    currentBestMatch = i;
                }
            }
        }
        if (!bFindFps) {
            if (currentBestMatch >= 0) {
                framerateRange[0] = frameRates.data.i32[currentBestMatch * 2 + 0];
                framerateRange[1] = frameRates.data.i32[currentBestMatch * 2 + 1];
            } else {
                framerateRange[0] = frameRates.data.i32[0];
                framerateRange[1] = frameRates.data.i32[1];
            }
        }
    }
    // 查找前后置
    camera_status_t status =
        ACameraMetadata_getConstEntry(metadata, ACAMERA_LENS_FACING, &entry);
    if (status == ACAMERA_OK) {
        auto facing = static_cast<acamera_metadata_enum_android_lens_facing_t>(
            entry.data.u8[0]);
        isBack = facing == ACAMERA_LENS_FACING_BACK;
    }
    // 默认选择第一个输出格式
    setFormat(0);
    return true;
}

void AVideoDevice::onClose() {
    if (session) {
        ACameraCaptureSession_stopRepeating(session);
        ACameraCaptureSession_close(session);
        session = nullptr;
    }
    if (request) {
        ACaptureRequest_removeTarget(request, outputTarget);
        ACaptureRequest_free(request);
        ACameraOutputTarget_free(outputTarget);
        request = nullptr;
        outputTarget = nullptr;
    }
    if (sessionOutput) {
        ACaptureSessionOutputContainer_remove(outputContainer, sessionOutput);
        ACaptureSessionOutput_free(sessionOutput);
        sessionOutput = nullptr;
    }
    if (surface) {
        ANativeWindow_release(surface);
        surface = nullptr;
    }
    if (outputContainer) {
        ACaptureSessionOutputContainer_free(outputContainer);
        outputContainer = nullptr;
    }
    if (ndkDevice) {
        ACameraDevice_close(ndkDevice);
        ndkDevice = nullptr;
    }
    if (imageReader) {
        AImageReader_delete(imageReader);
        imageReader = nullptr;
    }
    isOpen = false;
}

}  // namespace android
}  // namespace aoce