#pragma once
#include "../Layer/LayerFactory.hpp"
#include "../Layer/PipeGraph.hpp"

namespace aoce {

// 一个根据VideoFrame自动计算格式输出RGBA的图表
class ACOE_EXPORT VideoViewGraph {
   private:
    /* data */
    PipeGraph* graph = nullptr;
    InputLayer* inputLayer = nullptr;
    OutputLayer* outputLayer = nullptr;
    YUV2RGBALayer* yuv2rgbLayer = nullptr;
    ReSizeLayer* resizeLayer1 = nullptr;
    ReSizeLayer* resizelayer2 = nullptr;
#if WIN32
    GpuType gpuType = GpuType::cuda;
#elif __ANDROID__
    GpuType gpuType = GpuType::vulkan;
#endif
    VideoFormat format = {};

   public:
    VideoViewGraph(/* args */);
    ~VideoViewGraph();

   public:
    inline OutputLayer* getOutputLayer() { return outputLayer; }


   public:
    // (other根据平台自动选择)
    void initGraph(GpuType gpuType = GpuType::other);
    void updateParamet(YUVParamet yparamet, InputParamet iparamet = {},
                       OutputParamet oparamet = {});
    void runFrame(const VideoFrame& frame);
};

}  // namespace aoce