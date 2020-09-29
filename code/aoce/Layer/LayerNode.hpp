#pragma once

#include <vector>

#include "../Aoce.hpp"
#include "BaseLayer.hpp"

namespace aoce {

class ACOE_EXPORT LayerNode {
   private:
    /* data */
    friend class LayerGraph;
    struct ChildNode {
        int32_t inputIndex = 0;
        LayerNode* node = nullptr;
    };
    std::vector<ChildNode> nodes;
    BaseLayer* layer = nullptr;

   public:
    LayerNode(BaseLayer* _layer);
    virtual ~LayerNode();

   public:
    // 添加子节点,outputIndex表示本身输出节点索引,inputIndex表示本身输入索引
    LayerNode* addNode(BaseLayer* layer, int32_t outputIndex = 0,
                       int32_t inputIndex = 0);
};
}  // namespace aoce