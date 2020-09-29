#pragma once

#include "../Aoce.hpp"
namespace aoce {

// 当前层支持的GPU类型
enum class GpuBit {
    other = 0,
    vulkan = 1,
    dx11 = 2,
    cuda = 4,
};

class ACOE_EXPORT BaseLayer {
   private:
    friend class LayerNode;

    // 定义当前层需要的输入数量
    int32_t inputCount = 1;
    // 定义当前层需要的输出数量
    int32_t outputCount = 1;

   public:
    BaseLayer(/* args */);
    virtual ~BaseLayer();
};

}  // namespace aoce