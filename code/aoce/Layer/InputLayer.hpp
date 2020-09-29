#pragma once
#include "BaseLayer.hpp"

namespace aoce {
// inputlayer 应该从VkBaseLayer/Dx11BaseLayer/CudaBaseLayer继承
class ACOE_EXPORT InputLayer : public virtual BaseLayer {
   private:
    /* data */
   public:
    InputLayer(/* args */);
    virtual ~InputLayer();
};

}  // namespace aoce