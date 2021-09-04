#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkBlendingModeLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkDrawPointsPreLayer : public VkLayer, public ITLayer<PointsParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkDrawPointsPreLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
    std::unique_ptr<VulkanBuffer> inBuffer = nullptr;
    // std::unique_ptr<VulkanBuffer> inBufferX = nullptr;
    int32_t maxPoint = 108;

   public:
    VkDrawPointsPreLayer(/* args */);
    virtual ~VkDrawPointsPreLayer();

   public:
    void setImageFormat(const ImageFormat& imageFormat);
    void drawPoints(const vec2* points, int32_t size, vec4 color,
                    int32_t raduis);

   protected:
    virtual void onInitGraph() override;
    virtual void onInitVkBuffer() override;
    virtual void onInitPipe() override;
    virtual void onCommand() override;
};

class VkDrawPointsLayer : public VkLayer, public IDrawPointsLayer {
    AOCE_LAYER_QUERYINTERFACE(VkDrawPointsLayer)
   private:
    /* data */
    std::unique_ptr<VkDrawPointsPreLayer> preLayer = nullptr;

   public:
    VkDrawPointsLayer(/* args */);
    virtual ~VkDrawPointsLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
    //virtual void onCommand() override;

   public:
    virtual void drawPoints(const vec2* points, int32_t size, vec4 color,
                            int32_t raduis) override;
};

class VkDrawRectLayer : public VkLayer, public IDrawRectLayer {
    AOCE_LAYER_QUERYINTERFACE(VkDrawRectLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkDrawRectLayer(/* args */);
    virtual ~VkDrawRectLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce