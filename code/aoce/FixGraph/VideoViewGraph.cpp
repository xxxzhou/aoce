#include "VideoViewGraph.hpp"

#include "../AoceManager.hpp"
#include "../Layer/PipeNode.hpp"

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
    if (getYuvIndex(frame.videoType) >= 0) {
        yuv2rgbLayer->getNode()->setVisable(true);
        if (yuv2rgbLayer->getParamet().type != frame.videoType) {
            yuv2rgbLayer->updateParamet({frame.videoType});
        }
    } else {
        yuv2rgbLayer->getNode()->setVisable(false);
    }
    inputLayer->inputCpuData(frame, 0);
    graph->run();
}

}  // namespace aoce