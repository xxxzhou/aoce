#include "CuLayerFactory.hpp"

#include "layer/CuInputLayer.hpp"
#include "layer/CuOutputLayer.hpp"
#include "layer/CuRGBA2YUVLayer.hpp"
#include "layer/CuResizeLayer.hpp"
#include "layer/CuYUV2RGBALayer.hpp"

namespace aoce {
namespace cuda {

CuLayerFactory::CuLayerFactory(/* args */) {}

CuLayerFactory::~CuLayerFactory() {}

IInputLayer* CuLayerFactory::crateInput() { return new CuInputLayer(); }

IOutputLayer* CuLayerFactory::createOutput() { return new CuOutputLayer(); }

IYUV2RGBALayer* CuLayerFactory::createYUV2RGBA() {
    return new CuYUV2RGBALayer();
}

IRGBA2YUVLayer* CuLayerFactory::createRGBA2YUV() {
    return new CuRGBA2YUVLayer();
}
ITexOperateLayer* CuLayerFactory::createTexOperate() { return nullptr; }
ITransposeLayer* CuLayerFactory::createTranspose() { return nullptr; }
IReSizeLayer* CuLayerFactory::createSize() { return new CuResizeLayer(); }
IBlendLayer* CuLayerFactory::createBlend() { return nullptr; }

}  // namespace cuda
}  // namespace aoce
