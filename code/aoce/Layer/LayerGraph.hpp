#pragma once

#include "../Aoce.hpp"
#include "InputLayer.hpp"
#include "LayerNode.hpp"
namespace aoce {

// 参考 https://zhuanlan.zhihu.com/p/147207161
// 设计为有向无环图,node包含layer.
// node承担图像流程功能
// layer包含图像本身处理
class ACOE_EXPORT LayerGraph {
   private:
    /* data */
   public:
    LayerGraph(/* args */);
    virtual ~LayerGraph();

   public:
    LayerNode* addInputNode(InputLayer* layer);
};

}  // namespace aoce