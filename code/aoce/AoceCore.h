#pragma once
// 给外部用户使用

// 导出给外部用户使用,主要三种
// 1. C风格的结构,与C导出用来创建对应工厂/管理对象.
// 2. 纯净的抽像C++类,用户不要继承这些类,主要用来调用API.
// 3. 抽像C++类后缀Observer类的,用户继承针对接口处理回调.

#include "Aoce.h"
#include "AoceLayer.h"
#include "AoceLive.h"
#include "AoceMath.h"
#include "AoceMedia.h"
#include "AoceVideoDevice.h"

namespace aoce {

extern "C" {

// 检查模块aoce_cuda/aoce_vulkan/aoce_win_mf/aoce_ffmpeg是否加载成功
ACOE_EXPORT bool checkLoadModel(const char* modelName);

// 得到分配GPU管线的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT PipeGraphFactory* getPipeGraphFactory(const GpuType& gpuType);

// 得到分配基本层的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT LayerFactory* getLayerFactory(const GpuType& gpuType);

// 得到对应设备的管理对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT IVideoManager* getVideoManager(const CameraType& cameraType);

// 得到分配媒体播放器的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT MediaFactory* getMediaFactory(
    const MediaType& mediaType);

// 得到直播模块提供的管理对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT ILiveRoom* getLiveRoom(const LiveType& liveType);
}

}  // namespace aoce