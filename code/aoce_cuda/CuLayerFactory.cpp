#include "CuLayerFactory.hpp"

#include "layer/CuInputLayer.hpp"
#include "layer/CuOutputLayer.hpp"
#include "layer/CuYUV2RGBALayer.hpp"
#include "layer/CuRGBA2YUVLayer.hpp"

namespace aoce {
namespace cuda {

CuLayerFactory::CuLayerFactory(/* args */) {}

CuLayerFactory::~CuLayerFactory() {}

InputLayer* CuLayerFactory::crateInput() { return new CuInputLayer(); }

OutputLayer* CuLayerFactory::createOutput() { return new CuOutputLayer(); }

YUV2RGBALayer* CuLayerFactory::createYUV2RGBA() {
    return new CuYUV2RGBALayer();
}

RGBA2YUVLayer* CuLayerFactory::createRGBA2YUV() {
    return new CuRGBA2YUVLayer();
}
TexOperateLayer* CuLayerFactory::createTexOperate() {
    return nullptr;
}
TransposeLayer* CuLayerFactory::createTranspose() {
    return nullptr;
}
ReSizeLayer* CuLayerFactory::createSize() {
    return nullptr;
}
BlendLayer* CuLayerFactory::createBlend() {
    return nullptr;
}

}  // namespace cuda
}  // namespace aoce
