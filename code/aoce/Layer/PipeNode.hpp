#pragma once

#include <memory>
#include <vector>

#include "../Aoce.hpp"
#include "BaseLayer.hpp"

namespace aoce {

typedef std::shared_ptr<PipeNode> PipeNodePtr;

struct ChildNode {
    int32_t index = 0;
    PipeNodePtr node = nullptr;
};

// 请注意,使用BaseLayer初始化PipeNode时,相应的outputCount需要已经固定
// BaseLayer有几个输入,对应几个PipeNode,输出对应属性nodes
class ACOE_EXPORT PipeNode {
   private:
    /* data */
    friend class PipeGraph;
    std::vector<ChildNode> nodes;
    BaseLayer* layer = nullptr;
    // 如果为true,当前节点不使用
    bool bInvisible = false;
    // 如果为true,包含这个节点之后子节点不使用
    bool bDisable = false;
    int32_t graphIndex = 0;

   public:
    PipeNode(BaseLayer* _layer);
    virtual ~PipeNode();

   public:
    void setVisable(bool bvisable);
    void setEnable(bool benable);

   public:
    // 添加子节点,子节点的第childIndex输入节点连接本身节点的输出index索引上.
    // 其index代表layer里对应的输入索引,一个layer有多个输入就对应多个node
    PipeNodePtr addNode(BaseLayer* layer, int32_t childIndex = 0,
                        int32_t index = 0);
};
}  // namespace aoce