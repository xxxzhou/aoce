#include "BaseLayer.hpp"

namespace aoce {
BaseLayer::BaseLayer(/* args */) {}

BaseLayer::~BaseLayer() {}

void BaseLayer::init() {
    inputFormats.resize(inputCount);
    outputFormats.resize(outputCount);
    onInit();
}
}  // namespace aoce