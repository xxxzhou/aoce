#pragma once

#include <memory>
#include "aoce/AoceCore.h"

namespace aoce {

// 一个根据VideoFrame自动计算格式输出RGBA的图表
class VideoView {
   private:
    /* data */
    std::unique_ptr<IPipeGraph> graph = nullptr;
    std::unique_ptr<IInputLayer> inputLayer = nullptr;
    std::unique_ptr<IOutputLayer> outputLayer = nullptr;
    std::unique_ptr<IYUVLayer> yuv2rgbLayer = nullptr;

    GpuType gpuType = GpuType::other;
    VideoFormat format = {};

   public:
    VideoView(GpuType gpuType = GpuType::other);
    ~VideoView();

   public:
    inline IOutputLayer* getOutputLayer() { return outputLayer.get(); }

   public:
    void runFrame(const VideoFrame& frame, bool special = false);
};

}  // namespace aoce