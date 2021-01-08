#include "VideoViewGraph.hpp"

#include "../AoceManager.hpp"

namespace aoce {
VideoViewGraph::VideoViewGraph(/* args */) {}

VideoViewGraph::~VideoViewGraph() {}

void VideoViewGraph::initGraph(GpuType gpuType) {
    if (gpuType == GpuType::other) {
#if WIN32
        gpuType = GpuType::cuda;
#elif __ANDROID__
        gpuType = GpuType::vulkan;
#endif
    }
    // 生成一张执行图
    graph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto *layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    // 链接图
    graph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
}

void VideoViewGraph::updateParamet(YUVParamet yparamet, InputParamet iparamet,
                                   OutputParamet oparamet) {
    inputLayer->updateParamet(iparamet);
    yuv2rgbLayer->updateParamet(yparamet);
    outputLayer->updateParamet(oparamet);
}

void VideoViewGraph::runFrame(const VideoFrame &frame) {
    if (format.width != frame.width || format.height != frame.height ||
        format.videoType != frame.videoType) {
        format.width = frame.width;
        format.height = frame.height;
        format.videoType = frame.videoType;
        inputLayer->setImage(format);
        yuv2rgbLayer->updateParamet({format.videoType});
    }
    inputLayer->inputCpuData(frame, 0);
    graph->run();
}

}  // namespace aoce