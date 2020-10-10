#include "BaseLayer.hpp"

namespace aoce {
BaseLayer::BaseLayer(int32_t inSize, int32_t outSize) {
    inputCount = inSize;
    outputCount = outSize;
    inputFormats.resize(inputCount);
    outputFormats.resize(outputCount);
    inLayers.resize(inputCount);
    // 默认imagetype
    for (auto& format : inputFormats) {
        format.imageType = ImageType::rgba8;
    }
    for (auto& format : outputFormats) {
        format.imageType = ImageType::rgba8;
    }
}

BaseLayer::~BaseLayer() {}

bool BaseLayer::addInLayer(int32_t inIndex, int32_t nodeIndex,
                           int32_t outputIndex) {
    if (inIndex >= inputCount) {
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
    for (const auto& layer : inLayers) {
        if (layer.nodeIndex < 0 || layer.outputIndex < 0) {
            return false;
        }
        return true;
    }
}

void BaseLayer::initLayer() {
    int32_t size = inLayers.size();
    for (int32_t i = 0; i < size; i++) {
        // inputFormats[i] = pipeGraph // inLayers[i]
    }
    onInitLayer();
}

}  // namespace aoce