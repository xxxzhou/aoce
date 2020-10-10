#pragma once

#include <Layer/PipeGraph.hpp>

namespace aoce {
namespace vk {
namespace layer {

class VkPipeGraph : public PipeGraph {
   private:
    /* data */
   public:
    VkPipeGraph(/* args */);
    ~VkPipeGraph();

   public:
   public:
    // 限定不能直接创建
    virtual bool onRun();
};

class VkPipeGraphFactory : public PipeGraphFactory {
   public:
    VkPipeGraphFactory(){};
    virtual ~VkPipeGraphFactory(){};

   public:
    inline virtual PipeGraph* createGraph() override { return new VkPipeGraph(); };
};

}  // namespace layer
}  // namespace vk
}  // namespace aoce
