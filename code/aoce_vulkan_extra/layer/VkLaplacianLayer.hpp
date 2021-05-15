#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 卷积内核是一个3x3的值矩阵,适用于该像素及其周围的8个像素.
// 矩阵按行优先顺序指定,左上像素为one.one,右下像素为three.three.
// 如果矩阵中的值之和不等于1.0,则图像可能变亮或变暗
class Vk3x3ConvolutionLayer : public VkLayer, public ITLayer<Mat3x3> {
    AOCE_LAYER_QUERYINTERFACE(Vk3x3ConvolutionLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   public:
    // small 中间-4,否则-8,边缘更深
    Vk3x3ConvolutionLayer();
    ~Vk3x3ConvolutionLayer();
};

// 构造是传入false,更多的细节边.
class VkLaplacianLayer : public VkLayer {
   private:
    /* data */
    Mat3x3 mat = {};

   public:
    // small 中间-4,否则-8,边缘更深
    VkLaplacianLayer(bool small);
    ~VkLaplacianLayer();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce