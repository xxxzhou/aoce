#pragma once

#include <Layer/LayerFactory.hpp>

namespace aoce {
namespace cuda {

class CuLayerFactory : public LayerFactory {
   private:
    /* data */
   public:
    CuLayerFactory(/* args */);
    virtual ~CuLayerFactory() override;

   public:
    virtual InputLayer* crateInput() override;
    virtual OutputLayer* createOutput() override;
    virtual YUV2RGBALayer* createYUV2RGBA() override;
};

}  // namespace cuda
}  // namespace aoce
