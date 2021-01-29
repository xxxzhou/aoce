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
        layer->pipeGraph->reset();
    }
}

void PipeNode::setEnable(bool benable) {
    if (bDisable == benable) {
        bDisable = !benable;
        layer->pipeGraph->reset();
    }
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
    layer->pipeGraph->addLine(this->graphIndex, to->graphIndex, formOut, toIn);
    return to;
}

}  // namespace aoce