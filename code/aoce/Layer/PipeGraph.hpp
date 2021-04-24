#pragma once

#include <list>
#include <memory>
#include <vector>

#include "../Aoce.hpp"
#include "InputLayer.hpp"
#include "PipeNode.hpp"
#include <mutex>

namespace aoce {

// 参考 https://zhuanlan.zhihu.com/p/147207161
// 设计为有向无环图,node包含layer.
// node承担图像流程功能
// layer包含图像本身处理
class ACOE_EXPORT PipeGraph {
   public:
    PipeGraph(/* args */);
    virtual ~PipeGraph();

   private:
    bool checkHaveValid(PipeLinePtr ptr);
    // friend class BaseLayer;    
    void validNode();

   protected:
    GpuType gpu = GpuType::other;
    std::vector<PipeLinePtr> lines;
    std::vector<PipeLinePtr> validLines;
    std::vector<PipeNodePtr> nodes;
    // 需要重新reset.
    bool bReset = false;    
    // 图表的执行顺序
    std::vector<int32_t> nodeExcs;
    std::mutex mtx;

   protected:
    // 重新构建有序无环图的执行顺序
    bool resetGraph();

   public:
    // 当前图使用的gpu类型
    inline GpuType getGpuType() { return gpu; }
    // 引发resetGraphr执行,但是不一定与当前执行同线程
    void reset() { bReset = true; }

    PipeNodePtr getNode(int32_t index);
    PipeNodePtr addNode(BaseLayer* layer);
    PipeNodePtr addNode(ILayer* layer);
    bool addLine(int32_t from, int32_t to, int32_t formOut = 0,
                 int32_t toIn = 0);
    bool addLine(PipeNodePtr from, PipeNodePtr to, int32_t formOut = 0,
                 int32_t toIn = 0);

    void getLayerOutFormat(int32_t nodeIndex, int32_t outputIndex,
                           ImageFormat& format,bool bOutput = false);

    // 清除连线(当逻辑变更导致执行列表重组)
    void clearLines();

    // 清除节点(需要重新变更整个逻辑)
    void clear();

   protected:
    // 如vulkan,需要同步资源
    virtual void onReset(){};
    virtual bool onInitLayers() { return false; };
    virtual bool onInitBuffers() { return false; }
    // 限定不能直接创建
    virtual bool onRun() = 0;

   public:
    bool run();
};

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class ACOE_EXPORT PipeGraphFactory {
   public:
    PipeGraphFactory(){};
    virtual ~PipeGraphFactory(){};

   public:
    virtual PipeGraph* createGraph() { return nullptr; };
};

}  // namespace aoce