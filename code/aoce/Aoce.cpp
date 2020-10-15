#include "Aoce.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "Aoce.hpp"
#include "Module/ModuleManager.hpp"
#if WIN32
#include <Windows.h>

#include <iomanip>
#elif __ANDROID__
#include <stdlib.h>
#define LOGI(...) \
    ((void)__android_log_print(ANDROID_LOG_INFO, "aoce", __VA_ARGS__))
#define LOGW(...) \
    ((void)__android_log_print(ANDROID_LOG_WARN, "aoce", __VA_ARGS__))
#define LOGE(...) \
    ((void)__android_log_print(ANDROID_LOG_ERROR, "aoce", __VA_ARGS__))
#define LOGD(...) \
    ((void)__android_log_print(ANDROID_LOG_DEBUG, "aoce", __VA_ARGS__))
#endif

using namespace aoce;

static logEventHandle logHandle = nullptr;

void setLogAction(logEventAction action) { logHandle = action; }

void logMessage(AOCE_LOG_LEVEL level, const char* message) {
#if !DEBUG
    if (level == AOCE_LOG_DEBUG) {
        return;
    }
#endif
    if (logHandle != nullptr) {
        logHandle(level, message);
    } else if (message) {
#if WIN32
        auto now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        struct tm t;
        localtime_s(&t, &now);
        // 用std::coute有可能会导致UE4烘陪失败,记录下
        std::wcout << std::put_time(&t, L"%Y-%m-%d %X") << " Level: " << level
                   << L" " << message << std::endl;
        // << "\" in " << __FILE__ << " at line " << __LINE__
#elif __ANDROID__
        switch (level) {
            case AOCE_LOG_INFO:
                LOGI(message, __FILE__, __LINE__);
                break;
            case AOCE_LOG_WARN:
                LOGW(message, __FILE__, __LINE__);
                break;
            case AOCE_LOG_ERROR:
                LOGE(message, __FILE__, __LINE__);
                break;
            case AOCE_LOG_DEBUG:
                LOGD(message, __FILE__, __LINE__);
                break;
            default:
                break;
        }
#endif
    }
}

void logMessage(aoce::LogLevel level, const std::string& message) {
    if (message.empty()) {
        return;
    }
    logMessage((AOCE_LOG_LEVEL)level, message.c_str());
}

long long getNowTimeStamp() {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto timeStamp =
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return timeStamp;
}

// ansi用char表示多字节编码(中国ansi对应GB2312),unicode(UTF-16)用wchar表示编码(多国家同一编码)
// uft8在unicode基础上,变长1-4Byte表示,用于传输与保存节约空间
// https://stackoverflow.com/questions/6693010/how-do-i-use-multibytetowidechar
// https://stackoverflow.com/questions/4804298/how-to-convert-wstring-into-string
std::wstring utf8TWstring(const std::string& str) {
    if (str.empty()) {
        return std::wstring();
    }
    size_t len = str.length() + 1;
    std::wstring ret = std::wstring(len, 0);
#if defined WIN32
    int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, &str[0],
                                   str.size(), &ret[0], len);
    ret.resize(size);
#else
    size_t size = 0;
    //_locale_t lc = _create_locale(LC_ALL, "en_US.UTF-8");
    errno_t retval = _mbstowcs_s(&size, &ret[0], len, &str[0], _TRUNCATE);
    //_free_locale(lc);
    ret.resize(size - 1);
#endif
    return ret;
}

std::string utf8TString(const std::wstring& wstr) {
    if (wstr.empty()) {
        return std::string();
    }
#if defined WIN32
    int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, &wstr[0],
                                   wstr.size(), NULL, 0, NULL, NULL);
    std::string ret = std::string(size, 0);
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, &wstr[0], wstr.size(),
                        &ret[0], size, NULL, NULL);
#else
    size_t size = 0;
    //_locale_t lc = _create_locale(LC_ALL, "en_US.UTF-8");
    errno_t err = _wcstombs_s(&size, NULL, 0, &wstr[0], _TRUNCATE);
    std::string ret = std::string(size, 0);
    err = _wcstombs_s(&size, &ret[0], size, &wstr[0], _TRUNCATE);
    //_free_locale(lc);
    ret.resize(size - 1);
#endif
    return ret;
}

void copywcharstr(wchar_t* dest, const wchar_t* source, int32_t maxlength) {
    int length = sizeof(wchar_t) * (wcslen(source) + 1);
    memcpy(dest, source, min(length, maxlength));
}

void copycharstr(char* dest, const char* source, int32_t maxlength) {
    int length = sizeof(char) * (strlen(source) + 1);
    memcpy(dest, source, min(length, maxlength));
}

uint32_t divUp(int32_t x, int32_t y) { return (x + y - 1) / y; }

aoce::ImageType videoType2ImageType(const aoce::VideoType& videoType) {
    switch (videoType) {
        case VideoType::nv12:
        case VideoType::yuv420P:
        case VideoType::yuy2P:
            return ImageType::r8;
        case VideoType::yuv2I:
        case VideoType::yvyuI:
        case VideoType::uyvyI:
        case VideoType::mjpg:
        case VideoType::rgba8:
            return ImageType::rgba8;
        // 这几个信息需要在InputLayer输出中转化成rgba
        case VideoType::argb8:
        case VideoType::rgb8:
        case VideoType::bgra8:
            return ImageType::rgba8;
        case VideoType::depth16u:
            return ImageType::r16;
        case VideoType::other:
        default:
            return ImageType::other;
    }
}

ImageFormat videoFormat2ImageFormat(const VideoFormat& videoFormat) {
    ImageFormat imageFormat = {};
    imageFormat.imageType = videoType2ImageType(videoFormat.videoType);
    imageFormat.width = videoFormat.width;
    imageFormat.height = videoFormat.height;
    if (videoFormat.videoType == VideoType::nv12 ||
        videoFormat.videoType == VideoType::yuv420P) {
        imageFormat.height = videoFormat.height * 3 / 2;
    } else if (videoFormat.videoType == VideoType::yuy2P) {
        imageFormat.height = videoFormat.height / 2;
    } else if (videoFormat.videoType == VideoType::uyvyI ||
               videoFormat.videoType == VideoType::yuv2I ||
               videoFormat.videoType == VideoType::yvyuI) {
        imageFormat.width = videoFormat.width / 2;
    }
    return imageFormat;
}

int32_t getImageTypeSize(const aoce::ImageType& imageType) {
    switch (imageType) {
        case ImageType::bgra8:
            return 4;
        case ImageType::r16:
            return 2;
        case ImageType::r8:
            return 1;
        case ImageType::rgba8:
            return 4;
        default:
            return 0;
    }
}

void loadAoce() {
#if WIN32
    ModuleManager::Get().regAndLoad("aoce_win_mf");
#endif
    ModuleManager::Get().regAndLoad("aoce_vulkan");
}

void unloadAoce() {
#if WIN32
    ModuleManager::Get().unloadModule("aoce_win_mf");
#endif
    ModuleManager::Get().unloadModule("aoce_vulkan");
}
