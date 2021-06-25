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
    markStr = "";
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

bool BaseLayer::bAttachGraph() { return !pipeNode.expired(); }

void BaseLayer::setVisable(bool bvisable) {
    cheackAttachGraph();
    getNode()->setVisable(bvisable);
}
void BaseLayer::setEnable(bool benable) {
    cheackAttachGraph();
    getNode()->setEnable(benable);
}
int32_t BaseLayer::getGraphIndex() {
    cheackAttachGraph();
    return getNode()->graphIndex;
}
void BaseLayer::setStartNode(IBaseLayer* node, int32_t index,
                             int32_t toInIndex) {
    cheackAttachGraph();
    return getNode()->setStartNode(node, index, toInIndex);
}
void BaseLayer::setEndNode(IBaseLayer* node) {
    cheackAttachGraph();
    return getNode()->setEndNode(node);
}
IBaseLayer* BaseLayer::addNode(IBaseLayer* layer) {
    cheackAttachGraph();
    IBaseLayer* ptr = pipeGraph->addNode(layer);
    return addLine(ptr, 0, 0);
}
IBaseLayer* BaseLayer::addNode(ILayer* layer) {
    return addNode(layer->getLayer());
}
IBaseLayer* BaseLayer::addLine(IBaseLayer* ito, int32_t formOut, int32_t toIn) {
    BaseLayer* to = static_cast<BaseLayer*>(ito);
    cheackAttachGraph();
    to->cheackAttachGraph();
    // 如果是输入层,不能使用addLine的方法给输入层加线,会构成回环
    if (!to->bInput) {
        assert(toIn < to->inCount);
        assert(formOut < outCount);
        int32_t toIndex = to->getGraphIndex();
        int32_t formIndex = getGraphIndex();
        // 添加逻辑,完善直接用有layer(endnode)->addline(layer)的节点添加
        if (getNode()->endNodeIndex >= 0) {
            formIndex = getNode()->endNodeIndex;
        }
        // 节点如果有多组输入toIn,一个输入可以有多个输出
        const auto& toStartNodes = to->getNode()->startNodes[toIn];
        if (toStartNodes.size() > 0) {
            for (const auto& startNode : toStartNodes) {
                pipeGraph->addLine(formIndex, startNode.nodeIndex, formOut,
                                   startNode.inIndex);
            }
        } else {
            pipeGraph->addLine(formIndex, toIndex, formOut, toIn);
        }
        if (to->getNode()->endNodeIndex >= 0) {
            IBaseLayer* result =
                pipeGraph->getNode(to->getNode()->endNodeIndex);
            assert(result);
            return result;
        }
    }
    return to;
}

void BaseLayer::cheackAttachGraph() {
    logAssert(!pipeNode.expired(), "please attach to the graph first.");
}

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

const char* BaseLayer::getName() { return "no define name"; }

const char* BaseLayer::getMark() {
    if (bAttachGraph()) {
        if (markStr.empty()) {
            string_format(markStr, "(", getGraphIndex(), ")", getName());
        }
        return markStr.c_str();
    }
    return getName();
}

GroupLayer::GroupLayer() { bNoCompute = true; }

GroupLayer::~GroupLayer() {}

// PipeNode* ILayer::getLayerNode() {
//     assert(getLayer());
//     return getLayer()->getNode().get();
// }

}  // namespace aoce