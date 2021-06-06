#include "CuLayerFactory.hpp"

#include "layer/CuComputeLayer.hpp"
#include "layer/CuInputLayer.hpp"
#include "layer/CuOutputLayer.hpp"
#include "layer/CuRGBA2YUVLayer.hpp"
#include "layer/CuResizeLayer.hpp"
#include "layer/CuYUV2RGBALayer.hpp"

namespace aoce {
namespace cuda {

CuLayerFactory::CuLayerFactory(/* args */) {}

CuLayerFactory::~CuLayerFactory() {}

IInputLayer* CuLayerFactory::createInput() { return new CuInputLayer(); }

IOutputLayer* CuLayerFactory::createOutput() { return new CuOutputLayer(); }

IYUVLayer* CuLayerFactory::createYUV2RGBA() {
    return new CuYUV2RGBALayer();
}

IYUVLayer* CuLayerFactory::createRGBA2YUV() {
    return new CuRGBA2YUVLayer();
}

IMapChannelLayer* CuLayerFactory::createMapChannel() {
    return new CuMapChannelLayer();
}

IFlipLayer* CuLayerFactory::createFlip() { return new CuFlipLayer(); }

ITransposeLayer* CuLayerFactory::createTranspose() {
    return new CuTransposeLayer();
}

IReSizeLayer* CuLayerFactory::createSize() { return new CuResizeLayer(); }

IBlendLayer* CuLayerFactory::createBlend() { return new CuBlendLayer(); }

}  // namespace cuda
}  // namespace aoce
