#include "BaseLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {
BaseLayer::BaseLayer(int32_t inSize, int32_t outSize) {
    inCount = inSize;
    outCount = outSize;
}

BaseLayer::~BaseLayer() {}

void BaseLayer::onInit() {
    inFormats.resize(inCount);
    outFormats.resize(outCount);
    inLayers.resize(inCount);
    // 默认imagetype
    for (auto& format : inFormats) {
        format.imageType = ImageType::rgba8;
    }
    for (auto& format : outFormats) {
        format.imageType = ImageType::rgba8;
    }
}

PipeGraph* BaseLayer::getGraph() { return pipeGraph; }

bool BaseLayer::addInLayer(int32_t inIndex, int32_t nodeIndex,
                           int32_t outputIndex) {
    if (inIndex >= inCount) {
        logMessage(LogLevel::warn, "layer add in layer error inindex.");
        return false;
    }
    if (inLayers[inIndex].nodeIndex >= 0) {
        logMessage(LogLevel::warn, "layer add in layer error have add.");
        return false;
    }
    inLayers[inIndex].nodeIndex = nodeIndex;
    inLayers[inIndex].outputIndex = outputIndex;
    return true;
}

bool BaseLayer::vaildInLayers() {
    // 输入层没有inLayers
    if (bInput) {
        return true;
    }
    for (const auto& layer : inLayers) {
        if (layer.nodeIndex < 0 || layer.outputIndex < 0) {
            return false;
        }
    }
    return true;
}

void BaseLayer::initLayer() {
    int32_t size = inLayers.size();
    if (!bInput) {
        for (int32_t i = 0; i < size; i++) {
            pipeGraph->getLayerOutFormat(inLayers[i].nodeIndex,
                                      inLayers[i].outputIndex, inFormats[i]);
        }
    }
    // 默认所有outputFormat == inputFormats[0]
    if (inFormats.size() > 0) {
        for (auto& outFormat : outFormats) {
            outFormat = inFormats[0];
        }
    }
    // 如果每层的outputFormat需要更新,请在如下函数单独处理
    onInitLayer();
}

}  // namespace aoce