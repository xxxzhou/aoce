#pragma once
#include <memory>
#include "../Layer/LayerFactory.hpp"
#include "../Layer/PipeGraph.hpp"

namespace aoce {

// 一个根据VideoFrame自动计算格式输出RGBA的图表
class ACOE_EXPORT VideoView {
   private:
    /* data */
    PipeGraph* graph = nullptr;
    std::unique_ptr<InputLayer> inputLayer = nullptr;
    std::unique_ptr<OutputLayer> outputLayer = nullptr;
    std::unique_ptr<YUV2RGBALayer> yuv2rgbLayer = nullptr;

    GpuType gpuType = GpuType::other;
    VideoFormat format = {};

   public:
    VideoView(GpuType gpuType = GpuType::other);
    ~VideoView();

   public:
    inline OutputLayer* getOutputLayer() { return outputLayer.get(); }

   public:
    void runFrame(const VideoFrame& frame, bool special = false);
};

}  // namespace aoce