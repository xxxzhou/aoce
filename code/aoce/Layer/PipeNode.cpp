#include "PipeNode.hpp"

#include "PipeGraph.hpp"
namespace aoce {
PipeNode::PipeNode(BaseLayer* _layer) {
    this->layer = layer;
    this->nodes.resize(layer->outputCount);
}

PipeNode::~PipeNode() {}

void PipeNode::setVisable(bool bvisable) {
    bInvisible = !bvisable;
    layer->pipeGraph->setReset();
}

void PipeNode::setEnable(bool benable) {
    bDisable = !benable;
    layer->pipeGraph->setReset();
}

PipeNodePtr PipeNode::addNode(BaseLayer* layer, int32_t childIndex,
                              int32_t index) {
    if (this->layer->gpu != layer->gpu) {
        logMessage(LogLevel::error, "layer gpu not equal node");
        return nullptr;
    }
    if (index >= this->nodes.size()) {
        logMessage(LogLevel::error, "index > nodesize");
        return nullptr;
    }
    PipeNodePtr cnode(new PipeNode(layer));
    nodes[index].node = cnode;
    nodes[index].index = childIndex;
    return cnode;
}

}  // namespace aoce