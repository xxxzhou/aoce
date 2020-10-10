#pragma once
#include "BaseLayer.hpp"

namespace aoce {

struct InputParamet {
    int32_t bCpu = true;
    int32_t bGpu = false;
};

// inputlayer 应该从VkBaseLayer/Dx11BaseLayer/CudaBaseLayer继承
class InputLayer : public ILayer<InputParamet> {
   public:
    virtual void setImage(ImageFormat imageFormat, int32_t index = 0) = 0;
};

}  // namespace aoce