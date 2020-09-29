#pragma once

//导出给外部库使用文件
#include "AoceDefine.h"

enum AOCE_LOG_LEVEL {
    AOCE_LOG_INFO,
    AOCE_LOG_WARN,
    AOCE_LOG_ERROR,
    // 在这定义只在debug下才会输出
    AOCE_LOG_DEBUG,
};

enum AOCE_GPU_SDK {
    AOCE_GPU_OTHER,
    AOCE_GPU_CUDA,
    AOCE_GPU_DX11,
    AOCE_GPU_VULKAN,
};

namespace aoce {
// 对应上面AOCE_LOG_LEVEL
enum class LogLevel {
    info = AOCE_LOG_INFO,
    warn = AOCE_LOG_WARN,
    error = AOCE_LOG_ERROR,
    debug = AOCE_LOG_DEBUG,
};

enum class GpuType {
    other = AOCE_GPU_OTHER,
    cuda = AOCE_GPU_CUDA,
    dx11 = AOCE_GPU_DX11,
    vulkan = AOCE_GPU_VULKAN,
};

// 视频设备与游戏内部纹理使用,YUV格式是Interleaved/Semi-Planar,没相到android设备直接读出p格式
// 难道android图像设备直接输出的是编码过的数据?还需要解码?等以后验证
enum class VideoType {
    other,
    // 这几种一般是window平台图像读取设备常用格式
    nv12,  // 420SP
    yuv2I,
    yvyuI,
    uyvyI,
    // MF设备,会自动解码成yuv2
    mjpg,
    rgb8,
    argb8,
    // 游戏常用,一般纹理与视频格式
    rgba8,
    // fbo,rtt常用
    bgra8,
    // 深度摄像头,16位uint
    depth16u,
    // 下面二种平面格式用于文件与传输,编解码常用,照说应单独分开,但android下设备读出是P格式,先合在一起看看
    yuy2P,
    yuv420P,
};

inline std::string to_string(VideoType value) {
    switch (value) {
        case VideoType::nv12:
            return "nv12";
        case VideoType::yuv2I:
            return "yuv2I";
        case VideoType::yvyuI:
            return "yvyuI";
        case VideoType::uyvyI:
            return "uyvyI";
        case VideoType::mjpg:
            return "mjpg";
        case VideoType::rgb8:
            return "rgb8";
        case VideoType::argb8:
            return "argb8";
        case VideoType::rgba8:
            return "rgba8";
        case VideoType::bgra8:
            return "bgra8";
        case VideoType::depth16u:
            return "depth16u";
        case VideoType::yuy2P:
            return "yuy2P";
        case VideoType::yuv420P:
            return "yuv420P";
        case VideoType::other:
        default:
            return "invalid";
    }
}

enum class CameraType {
    other,
    win_mf,
    and_camera2,
};

enum class ImageType {
    r8,
    r16,
    rgba8,
    bgra8,
};

enum class VideoCodec {
    other,
    h264,
};

enum class AudioCodec {
    other,
    acc,
};

// 视频设备输出格式
struct VideoFormat {
    int32_t index = -1;
    int32_t width = 0;
    int32_t height = 0;
    int32_t fps = 0;
    VideoType videoType = VideoType::other;
};

// 视频流,视频编解码需要的主要信息
struct VideoStream {
    VideoCodec codec = VideoCodec::h264;
    int32_t width = 0;
    int32_t height = 0;
    int32_t fps = 0;
    int32_t bitrate = 2000000;  // 2M
    // 应该只支持yuy2P/yuv420P这二种
    VideoType videoType = VideoType::other;
};

// 音频流,视频编解码需要的主要信息
struct AudioStream {
    AudioCodec codec = AudioCodec::acc;
    // 通道数
    int32_t channel = 1;
    int32_t sampleRate = 8000;  // 8000,11025,22050,44100
    int32_t depth = 16;         // 16,24,32
    int32_t bitrate = 48000;    // 48K
};

struct VideoFrame {
    int32_t width;
    int32_t height;
    int64_t timeStamp;
    VideoType videoType = VideoType::other;
    // 如果videoType非YUV平面格式,则只需填充data[0]里的空间
    uint8_t* data[4];
    // 大于等于width,满足整除32/16/8类型的值,同上非YUV平面格式,只有[0]里的值有意义
    int32_t dataAlign[4];
};

// 前期默认只支持单声道这一种格式,后期加上双通道
struct AudioFrame {
    // 默认只支持单/双二种通道格式
    int32_t channel = 1;
    int32_t sampleRate = 8000;
    int32_t depth = 16;
    int64_t timeStamp;
    // 先定义为frame里一个通道的byte长度
    int32_t dataSize;
    uint8_t* data[2];
};

}  // namespace aoce

extern "C" {

ACOE_EXPORT void setLogAction(logEventAction action);

ACOE_EXPORT void logMessage(AOCE_LOG_LEVEL level, const char* message);

ACOE_EXPORT long long getNowTimeStamp();
}