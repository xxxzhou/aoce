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
void PipeNode::setStartNode(BaseLayer* node, int32_t index, int32_t toInIndex) {
    if (toInIndex >= node->getNode()->startNodes.size()) {
        return;
    }
    // 自身为前置节点
    if (graphIndex == node->getGraphIndex()) {
        startNodes[index].push_back({node->getGraphIndex(), toInIndex});
        return;
    }
    const auto& nodes = node->getNode()->startNodes[toInIndex];
    // 如果当前组件也有头部
    if (nodes.size() > 0) {
        for (const auto& node : nodes) {
            BaseLayer* ptr = layer->getGraph()->getNode(node.nodeIndex);
            setStartNode(ptr, index, node.inIndex);
        }
    } else {
        startNodes[index].push_back({node->getGraphIndex(), toInIndex});
    }
}

void PipeNode::setEndNode(BaseLayer* node) {
    endNodeIndex = node->getGraphIndex();
}

}  // namespace aoce