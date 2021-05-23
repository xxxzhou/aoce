#pragma once

// 导出给外部库使用文件

#include <stdint.h>

#include "AoceDefine.h"

namespace aoce {

// 对应上面AOCE_LOG_LEVEL
enum class LogLevel : int32_t {
    info = 0,
    warn,
    error,
    debug,
};

// 完成win32端vulkan与dx11交互,
// 经测试性能与dx11性能还高一些,比不上cuda,不过作为通用备份选择足够了
enum class GpuType : int32_t {
    other = 0,
    cuda,
    vulkan,
};

// 视频设备与游戏内部纹理使用,YUV格式是Interleaved/Semi-Planar,没想到android设备直接读出p格式
// 难道android图像设备直接输出的是编码过的数据?还需要解码?等以后验证
enum class VideoType : int32_t {
    other = 0,
    // 这几种一般是window平台图像读取设备常用格式
    nv12,  // yuv420SP
    yuv2I,
    yvyuI,
    uyvyI,
    // MF设备,设定会自动解码成yuv2I
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

enum class CameraType : int32_t {
    other = 0,
    win_mf,
    and_camera2,
    realsense,
};

// aoce外部接收图像暂时就包含这几种
enum class ImageType : int32_t {
    other = 0,
    r8,
    rgba8,
    r16,
    // 游戏中fbo,rtt使用,输入与输出层使用,经输入层自动转化rgba8
    bgra8,
    rgba32f,
    r32f,
    r32,
    rgba32,
};

enum class VideoCodec : int32_t {
    other = 0,
    h264,
};

enum class AudioCodec : int32_t {
    other = 0,
    acc,
};

enum class LiveType : int32_t {
    other = 0,
    aoce,
    agora,
};

enum class MediaType : int32_t {
    other = 0,
    ffmpeg,
};

// 视频设备输出格式
struct VideoFormat {
    int32_t index = -1;
    int32_t width = 0;
    int32_t height = 0;
    int32_t fps = 0;
    VideoType videoType = VideoType::other;
};

struct ImageFormat {
    int32_t width = 0;
    int32_t height = 0;
    ImageType imageType = ImageType::other;

    inline bool operator==(const ImageFormat &right) {
        return this->width == right.width && this->height == right.height &&
               this->imageType == right.imageType;
    }
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
    uint8_t *data[4];
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
    uint8_t *data[2];
};

struct YUVParamet {
    VideoType type = VideoType::other;
    int32_t special = 0;
};

// ARGB<->BGRA<->RGBA<->RRRR
struct MapChannel {
    int32_t red = 0;
    int32_t green = 1;
    int32_t blue = 2;
    int32_t alpha = 3;
    inline bool operator==(const MapChannel &right) {
        return this->red == right.red && this->green == right.green &&
               this->blue == right.blue && this->alpha == right.alpha;
    }
};

// 方便C#交互不做额外设置,以及GPU参数结构对应,bool全用int表示
struct Operate {
    // 是否X倒转
    int32_t bFlipX = false;
    int32_t bFlipY = false;
    // 调整亮度
    float gamma = 1.f;
    inline bool operator==(const Operate &right) {
        return this->bFlipX == right.bFlipX && this->bFlipY == right.bFlipY &&
               this->gamma == right.gamma;
    }
};

// 纹理些基本操作
struct TexOperateParamet {
    MapChannel mapChannel = {};
    Operate operate = {};
    inline bool operator==(const TexOperateParamet &right) {
        return this->mapChannel == right.mapChannel &&
               this->operate == right.operate;
    }
};

// 纹理混合
struct BlendParamet {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float width = 0.4f;
    float height = 0.4f;
    // 显示如上位置图像的透明度
    float alaph = 0.2f;

    inline bool operator==(const BlendParamet &right) {
        return this->centerX == right.centerX &&
               this->centerY == right.centerY && this->width == right.width &&
               this->height == right.height && this->alaph == right.alaph;
    }
};

// width/height 变成height/width
struct TransposeParamet {
    // 是否X倒转
    int32_t bFlipX = false;
    int32_t bFlipY = false;
    inline bool operator==(const TransposeParamet &right) {
        return this->bFlipX == right.bFlipX && this->bFlipY == right.bFlipY;
    }
};

struct ReSizeParamet {
    int32_t bLinear = 1;
    int32_t newWidth = 0;
    int32_t newHeight = 0;
    inline bool operator==(const ReSizeParamet &right) {
        return this->newWidth == right.newWidth &&
               this->newHeight == right.newHeight &&
               this->bLinear == right.bLinear;
    }
};

struct SizeScaleParamet {
    int32_t bLinear = 1;
    float fx = 1.f;
    float fy = 1.f;
    inline bool operator==(const SizeScaleParamet &right) {
        return this->fx == right.fx && this->fy == right.fy &&
               this->bLinear == right.bLinear;
    }
};

//日志回调
typedef void (*logEventAction)(int32_t level, const char *message);

extern "C" {

ACOE_EXPORT void setLogAction(logEventAction action);

ACOE_EXPORT const char *getLogLevel(LogLevel level);

ACOE_EXPORT void logMessage(LogLevel level, const char *message);

ACOE_EXPORT const char *to_string(const VideoType &value);

ACOE_EXPORT uint32_t divUp(int32_t x, int32_t y);
ACOE_EXPORT long long getNowTimeStamp();

ACOE_EXPORT aoce::ImageType videoType2ImageType(
    const aoce::VideoType &videoType);

// 原则上,应该只由VideoType转ImageType
// ImageType转VideoType,只有bgra8/r16/rgba8三种有意义
ACOE_EXPORT aoce::VideoType imageType2VideoType(
    const aoce::ImageType &imageType);

ACOE_EXPORT int32_t getYuvIndex(const aoce::VideoType &videoType);

ACOE_EXPORT aoce::ImageFormat videoFormat2ImageFormat(
    const aoce::VideoFormat &videoFormat);

ACOE_EXPORT int32_t getImageTypeSize(const aoce::ImageType &imageType);

// 平面格式可能非紧密排列,给GPU的紧密排列大小,否则返回0
ACOE_EXPORT int32_t getVideoFrame(const aoce::VideoFrame &frame,
                                  uint8_t *data = nullptr);

ACOE_EXPORT void loadAoce();
ACOE_EXPORT void unloadAoce();

#if __ANDROID__
// ACOE_EXPORT jint JNI_OnLoad(JavaVM *jvm, void *);
// ACOE_EXPORT void JNI_OnUnload(JavaVM *jvm, void *);
struct AndroidEnv {
    JavaVM *vm = nullptr;
    // 在调用initAndroid里线程,注意不同线程这值不同
    JNIEnv *env = nullptr;
    jobject activity = nullptr;
    jobject application = nullptr;
    int32_t sdkVersion = 0;
    AAssetManager *assetManager = nullptr;
};
ACOE_EXPORT void initAndroid(const AndroidEnv &andEnv);

#endif
}

}  // namespace aoce
