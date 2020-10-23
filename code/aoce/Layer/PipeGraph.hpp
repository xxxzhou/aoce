#pragma once

#include <list>
#include <memory>
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
   public:
    PipeGraph(/* args */);
    virtual ~PipeGraph();

   private:
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

   protected:
    // 重新构建有序无环图的执行顺序
    bool resetGraph();

   public:
    // 当前图使用的gpu类型
    inline GpuType getGpuType() { return gpu; }
    void reset() { bReset = true; }

    PipeNodePtr addNode(BaseLayer* layer);
    PipeNodePtr addNode(ILayer* layer);
    bool addLine(int32_t from, int32_t to, int32_t formOut = 0,
                 int32_t toIn = 0);
    bool addLine(PipeNodePtr from, PipeNodePtr to, int32_t formOut = 0,
                  int32_t toIn = 0);

    void getImageFormat(int32_t nodeIndex, int32_t outputIndex,
                        ImageFormat& format);

   protected:
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