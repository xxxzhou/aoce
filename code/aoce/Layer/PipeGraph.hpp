#pragma once

#include <vector>

#include "../Aoce.hpp"
#include "InputLayer.hpp"
#include "PipeNode.hpp"
namespace aoce {

// 参考 https://zhuanlan.zhihu.com/p/147207161
// 设计为有向无环图,node包含layer.
// node承担图像流程功能
// layer包含图像本身处理
class ACOE_EXPORT PipeGraph {
   private:
    /* data */
    GpuType gpu = GpuType::other;
    std::vector<PipeNodePtr> inputNodes;
    // 需要重新reset.
    bool bReset = false;

   public:
    PipeGraph(/* args */);
    virtual ~PipeGraph();
    // 当前图使用的gpu类型
    inline GpuType getGpuType() { return gpu; }
    PipeNodePtr addInputNode(InputLayer* layer);
    void setReset() { bReset = true; }

   private:
    bool resetGraph(const PipeNodePtr& node);

   public:
    // 限定不能直接创建
    virtual bool onRun() = 0;

   public:
    // 把图的结构变成线性执行结构
    bool resetGraph();
    bool run();
};

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class ACOE_EXPORT PipeGraphFactory {
   public:
    PipeGraphFactory(){};
    virtual ~PipeGraphFactory(){};

   public:
    PipeGraph* createGraph() { return nullptr; };
};

}  // namespace aoce