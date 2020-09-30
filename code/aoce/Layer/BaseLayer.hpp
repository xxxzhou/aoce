#pragma once

#include <vector>

#include "../Aoce.hpp"
namespace aoce {

// 当前层支持的GPU类型
enum class GpuBit {
    other = 0,
    vulkan = 1,
    dx11 = 2,
    cuda = 4,
};

// BaseLayer定义可以在外部new,这样可以外接插件只管处理逻辑
// layer知道自己的gpu类型.设计分为二种
// 其一是本身插件提供公共处理,配合factory.
// 其二是外部插件提供new obj,需要自己设定gpu类型
class ACOE_EXPORT BaseLayer {
   protected:
    friend class PipeNode;
    friend class PipeGraph;

    GpuType gpu = GpuType::other;
    // 定义当前层需要的输入数量
    int32_t inputCount = 1;
    // 定义当前层需要的输出数量
    int32_t outputCount = 1;
    class PipeGraph* pipeGraph = nullptr;
    // 每个层的imagetype对应shader里类型,连接时需要检测
    std::vector<ImageFormat> inputFormats;
    // 每个层的imagetype对应shader里类型,连接时需要检测
    std::vector<ImageFormat> outputFormats;

   public:
    BaseLayer(/* args */);
    virtual ~BaseLayer();

   protected:
    // 根据inputCount/outputCount做资源初始化,inputFormats/outputFormats确定imagetype
    virtual void onInit(){};
    // 已经添加进pipeGraph,pipeGraph把所有层连接起来,此时知道inputFormats的长宽
    // 并根据当前层的需求,设定对应outFormats,也就是下一层的inputFormats
    // 可分配线程组的大小了
    virtual void onInitLayer(){};
    // 根据inputFormats初始化buffer
    virtual void onInitBuffer(){};
    // 更新参数,子类会有updateParamet(T t)保存参数,等到运行前提交执行
    virtual void onUpdateParamet(){};
    virtual void onRun(){};

   public:
    void init();
};

}  // namespace aoce