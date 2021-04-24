#include "PipeNode.hpp"

#include "PipeGraph.hpp"
namespace aoce {

PipeNode::PipeNode(BaseLayer* _layer) {
    if (_layer == nullptr) {
        logMessage(LogLevel::error, "node layer can not be empty");
    }
    assert(_layer != nullptr);
    this->layer = _layer;
    endNodeIndex = -1;
    startNodes.resize(this->layer->inCount);
}

PipeNode::~PipeNode() { startNodes.clear(); }

void PipeNode::setVisable(bool bvisable) {
    if (bInvisible == bvisable) {
        bInvisible = !bvisable;
        layer->resetGraph();
    }
}

void PipeNode::setEnable(bool benable) {
    if (bDisable == benable) {
        bDisable = !benable;
        layer->resetGraph();
    }
}

// index表示输入节点索引,node表示层内层节点,toInIndex表示对应层内层输入位置
// 简单来说,index对应本身node,toInIndex对应toNode
void PipeNode::setStartNode(PipeNodePtr node, int32_t index,
                            int32_t toInIndex) {
    if (toInIndex >= node->startNodes.size()) {
        return;
    }
    // 自身为前置节点
    if (this->getNodeIndex() == node->getNodeIndex()) {
        startNodes[index].push_back({node->getNodeIndex(), toInIndex});
        return;
    }
    const auto& nodes = node->startNodes[toInIndex];
    // 如果当前组件也有头部
    if (nodes.size() > 0) {
        for (const auto& node : nodes) {
            PipeNodePtr ptr = layer->getGraph()->getNode(node.nodeIndex);
            setStartNode(ptr, index, node.inIndex);
        }
    } else {
        startNodes[index].push_back({node->getNodeIndex(), toInIndex});
    }
}

void PipeNode::setEndNode(PipeNodePtr node) {
    endNodeIndex = node->getNodeIndex();
}

PipeNodePtr PipeNode::addNode(BaseLayer* layer) {
    if (this->layer->gpu != layer->gpu) {
        logMessage(LogLevel::error, "layer gpu not equal node");
        // return nullptr;
    }
    PipeNodePtr ptr = this->layer->pipeGraph->addNode(layer);
    return addLine(ptr, 0, 0);
}

PipeNodePtr PipeNode::addNode(ILayer* layer) {
    assert(layer != nullptr);
    return addNode(layer->getLayer());
}

PipeNodePtr PipeNode::addLine(PipeNodePtr to, int32_t formOut, int32_t toIn) {
    assert(toIn < to->layer->inCount);
    assert(formOut < layer->outCount);
    int toIndex = to->graphIndex;
    // 节点如果有多组输入toIn,一个输入可以有多个输出
    if (to->startNodes[toIn].size() > 0) {
        for (const auto& startNode : to->startNodes[toIn]) {
            layer->pipeGraph->addLine(this->graphIndex, startNode.nodeIndex,
                                      formOut, startNode.inIndex);
        }
    } else {
        layer->pipeGraph->addLine(this->graphIndex, toIndex, formOut, toIn);
    }
    if (to->endNodeIndex >= 0) {        
        PipeNodePtr result = layer->getGraph()->getNode(to->endNodeIndex);
        assert(result);
        return result;
    }
    return to;
}

}  // namespace aoce