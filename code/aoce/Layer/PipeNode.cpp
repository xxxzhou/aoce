#include "PipeNode.hpp"

#include "PipeGraph.hpp"
namespace aoce {

PipeNode::PipeNode(BaseLayer* _layer) {
    if (_layer == nullptr) {
        logMessage(LogLevel::error, "node layer can not be empty");
    }
    assert(_layer != nullptr);
    this->layer = _layer;
}

PipeNode::~PipeNode() {}

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

void PipeNode::setStartNode(PipeNodePtr node) { startNode = node; }
void PipeNode::setEndNode(PipeNodePtr node) { endNode = node; }

PipeNodePtr PipeNode::getStartNode() {
    if (startNode.expired()) {
        return nullptr;
    }
    return startNode.lock();
}

PipeNodePtr PipeNode::getEndNode() {
    if (endNode.expired()) {
        return nullptr;
    }
    return endNode.lock();
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
    int toIndex = to->graphIndex;
    auto start = to->getStartNode();
    if (start) {
        toIndex = start->graphIndex;
    }
    layer->pipeGraph->addLine(this->graphIndex, toIndex, formOut, toIn);
    auto end = to->getEndNode();
    if (end) {
        return end;
    }
    return to;
}

}  // namespace aoce