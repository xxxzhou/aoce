#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 像素图效果,马赛克
class VkPixellateLayer : public VkLayer, public ITLayer<PixellateParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkPixellateLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkPixellateLayer(/* args */);
    virtual ~VkPixellateLayer();
};

// 半色调效果，如新闻打印
class VkHalftoneLayer : public VkLayer, public ITLayer<PixellateParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkHalftoneLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkHalftoneLayer(/* args */);
    virtual ~VkHalftoneLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce