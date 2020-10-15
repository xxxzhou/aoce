#include "InputLayer.hpp"

#include "PipeGraph.hpp"
namespace aoce {

void InputLayer::setImage(VideoFormat videoFormat, int32_t index ) {
    assert(this->getLayer() != nullptr);
    assert(this->getLayer()->getGraph() != nullptr);

    onSetImage(videoFormat, index);
    // 重新组织图
    this->getLayer()->getGraph()->reset();
}

void InputLayer::inputCpuData(uint8_t* data, int32_t index) {
    onInputCpuData(data, index);    
}

}  // namespace aoce