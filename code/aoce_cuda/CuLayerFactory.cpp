#include "CuLayerFactory.hpp"

#include "layer/CuInputLayer.hpp"
#include "layer/CuOutputLayer.hpp"
#include "layer/CuYUV2RGBALayer.hpp"

namespace aoce {
namespace cuda {

CuLayerFactory::CuLayerFactory(/* args */) {}

CuLayerFactory::~CuLayerFactory() {}

InputLayer* CuLayerFactory::crateInput() { return new CuInputLayer(); }

OutputLayer* CuLayerFactory::createOutput() { return new CuOutputLayer(); }

YUV2RGBALayer* CuLayerFactory::createYUV2RGBA() {
    return new CuYUV2RGBALayer();
}

}  // namespace cuda
}  // namespace aoce
