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
    struct PipeLine {
       public:
        int32_t fromNode = -1;
        int32_t fromOutIndex = -1;
        int32_t toNode = -1;
        int32_t toInIndex = -1;

        bool bOutput() { return toNode < 0; }

        inline bool operator==(const PipeLine& right) {
            return this->fromNode == right.fromNode &&
                   this->fromOutIndex == right.fromOutIndex &&
                   this->toNode == right.toNode &&
                   this->toInIndex == right.toInIndex;
        }
    };
    std::vector<PipeLine> lines;

   private:
    GpuType gpu = GpuType::other;
    std::vector<PipeNodePtr> nodes;
    // 需要重新reset.
    bool bReset = false;
    // 图表的执行顺序
    std::vector<int32_t> nodeExcs;

   public:
    PipeGraph(/* args */);
    virtual ~PipeGraph();
    // 当前图使用的gpu类型
    inline GpuType getGpuType() { return gpu; }
    void setReset() { bReset = true; }

    PipeNodePtr addNode(BaseLayer* layer);
    void addLine(PipeNodePtr from, PipeNodePtr to, int32_t formOut = 0,
                 int32_t toIn = 0);

   public:
    // 限定不能直接创建
    virtual bool onRun() = 0;

   public:
    // 重新构建有序无环图的执行顺序
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