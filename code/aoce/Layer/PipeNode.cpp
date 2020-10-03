#include "PipeNode.hpp"

#include "PipeGraph.hpp"
namespace aoce {
PipeNode::PipeNode(BaseLayer* _layer) { this->layer = layer; }

PipeNode::~PipeNode() {}

void PipeNode::setVisable(bool bvisable) {
    bInvisible = !bvisable;
    layer->pipeGraph->setReset();
}

void PipeNode::setEnable(bool benable) {
    bDisable = !benable;
    layer->pipeGraph->setReset();
}

PipeNodePtr PipeNode::addNode(BaseLayer* layer) {
    if (this->layer->gpu != layer->gpu) {
        logMessage(LogLevel::error, "layer gpu not equal node");
        return nullptr;
    }
    PipeNodePtr ptr = this->layer->pipeGraph->addNode(layer);
    return addLine(ptr,0,0);    
}

PipeNodePtr PipeNode::addLine(PipeNodePtr to, int32_t formOut, int32_t toIn) {
    this->layer->pipeGraph->addLine(std::shared_ptr<PipeNode>(this), to, formOut, toIn);
    return to;
}

}  // namespace aoce