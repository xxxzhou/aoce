#pragma once

#include "CNNHelper.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

using namespace aoce::vulkan::layer;

namespace aoce {

class VkNcnnInLayer : public VkLayer, public ITLayer<NcnnInParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkNcnnInLayer)
   private:
    /* data */
   public:
    VkNcnnInLayer(/* args */);
    ~VkNcnnInLayer();

   protected:
    virtual void onUpdateParamet() override;
};

}  // namespace aoce