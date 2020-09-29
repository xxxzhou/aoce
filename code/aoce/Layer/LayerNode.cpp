#include "LayerNode.hpp"

namespace aoce {
LayerNode::LayerNode(BaseLayer* _layer) { this->layer = layer; }

LayerNode::~LayerNode() {}

LayerNode* LayerNode::addNode(BaseLayer* layer, int32_t outputIndex,
                              int32_t inputIndex) {
    if (outputIndex >= nodes.size()) {
        nodes.resize(outputIndex + 1);
    }
    LayerNode* childNode = new LayerNode(layer);
    nodes[outputIndex].node = childNode;
    nodes[outputIndex].inputIndex = inputIndex;
    return childNode;
}

}  // namespace aoce