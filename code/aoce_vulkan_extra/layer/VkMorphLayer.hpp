#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 膨胀与腐蚀(Dilation and Erosion)
// 膨胀和腐蚀被称为形态学操作。它们通常在二进制图像上执行，类似于轮廓检测。通过将像素添加到该图像中的对象的感知边界，扩张放大图像中的明亮白色区域。侵蚀恰恰相反：它沿着物体边界移除像素并缩小物体的大小。
// Closing 先dilation后erosion

class VkPreDilationLayer : public VkLayer, public ITLayer<int32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkPreDilationLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkPreDilationLayer();
    virtual ~VkPreDilationLayer();

   protected:
    virtual void onInitGraph() override;
};

// 扩张放大图像中的明亮白色区域
class VkDilationLayer : public VkLayer, public ITLayer<int32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkDilationLayer)
   public:
    VkDilationLayer();
    virtual ~VkDilationLayer();

   protected:
    std::unique_ptr<VkPreDilationLayer> preLayer = nullptr;

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

class VkPreErosionLayer : public VkLayer, public ITLayer<int32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkPreErosionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    VkPreErosionLayer();
    virtual ~VkPreErosionLayer();

   protected:
    virtual void onInitGraph() override;
};

// 它沿着物体边界移除像素并缩小物体的明亮白色区域
class VkErosionLayer : public VkLayer, public ITLayer<int32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkErosionLayer)
   public:
    VkErosionLayer();
    virtual ~VkErosionLayer();

   protected:
    std::unique_ptr<VkPreErosionLayer> preLayer = nullptr;

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

// Closing 先dilation后erosion
class VkClosingLayer : public GroupLayer, public ITLayer<int32_t> {
    AOCE_LAYER_QUERYINTERFACE(VkClosingLayer)
   public:
    VkClosingLayer();
    virtual ~VkClosingLayer();

   protected:
    std::unique_ptr<VkDilationLayer> dilationLayer = nullptr;
    std::unique_ptr<VkErosionLayer> erosionLayer = nullptr;

   protected:
    virtual void onUpdateParamet() override;

    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce