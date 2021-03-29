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
    outLayers.resize(outCount);    
    // 默认imagetype
    for (auto& format : inFormats) {
        format.imageType = ImageType::rgba8;
    }
    for (auto& format : outFormats) {
        format.imageType = ImageType::rgba8;
    }
}

PipeGraph* BaseLayer::getGraph() { return pipeGraph; }

PipeNodePtr BaseLayer::getNode() {
    if (pipeNode.expired()) {
        return nullptr;
    }
    return pipeNode.lock();
}

int32_t BaseLayer::getInCount() { return inCount; };
int32_t BaseLayer::getOutCount() { return outCount; };

bool BaseLayer::addInLayer(int32_t inIndex, int32_t nodeIndex,
                           int32_t outputIndex) {
    if (inIndex >= inCount) {
        logMessage(LogLevel::warn, "layer add in layer error inindex.");
        return false;
    }
    inLayers[inIndex].nodeIndex = nodeIndex;
    inLayers[inIndex].siteIndex = outputIndex;
    return true;
}

bool BaseLayer::addOutLayer(int32_t outIndex, int32_t nodeIndex,
                            int32_t inIndex) {
    if (outIndex >= outCount) {
        logMessage(LogLevel::warn, "layer add in layer error inindex.");
        return false;
    }
    NodeIndex nd = {nodeIndex, inIndex};
    outLayers[outIndex].push_back(nd);
    return true;
}

bool BaseLayer::vaildInLayers() {
    // 输入层没有inLayers
    if (bInput) {
        return true;
    }
    for (const auto& layer : inLayers) {
        if (layer.nodeIndex < 0 || layer.siteIndex < 0) {
            return false;
        }
    }
    return true;
}

void BaseLayer::initLayer() {
    int32_t size = inLayers.size();
    // 拿到上层的长宽
    if (!bInput) {
        for (int32_t i = 0; i < size; i++) {
            pipeGraph->getLayerOutFormat(inLayers[i].nodeIndex,
                                         inLayers[i].siteIndex, inFormats[i],
                                         bOutput || bAutoImageType);
        }
    }
    // 默认所有输出长宽为第一个输入
    if (inFormats.size() > 0) {
        for (auto& outFormat : outFormats) {
            outFormat.width = inFormats[0].width;
            outFormat.height = inFormats[0].height;
            if (bOutput) {
                outFormat.imageType = inFormats[0].imageType;
            }
        }
    }
    // 如果每层的outputFormat需要更新,请在如下函数单独处理
    onInitLayer();
}

void BaseLayer::resetGraph() {
    if (pipeGraph) {
        pipeGraph->reset();
    }
}

bool BaseLayer::getInFormat(ImageFormat& format, int32_t index) {
    if (inFormats.size() > index) {
        format = inFormats[index];
        return true;
    }
    return false;
}

PipeNode* ILayer::getLayerNode() {
    assert(getLayer());
    return getLayer()->getNode().get();
}

}  // namespace aoce