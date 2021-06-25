#pragma once

#include <vector>

#include "../Aoce.hpp"
namespace aoce {

// 当前层支持的GPU类型
enum class GpuBit {
    other = 0,
    vulkan = 1,
    cuda = 2,
    // dx11 = 4,
};

#define AOCE_LAYER_GETNAME(CLASS) \
   public:                        \
    virtual const char* getName() override { return #CLASS; }

// 每个从继承ILayer的类,请在类头文件里添加这个宏,或是自己实现
#define AOCE_LAYER_QUERYINTERFACE(OBJCLASS)           \
   public:                                            \
    virtual inline IBaseLayer* getLayer() override {  \
        OBJCLASS* obj = static_cast<OBJCLASS*>(this); \
        return static_cast<BaseLayer*>(obj);          \
    }                                                 \
    AOCE_LAYER_GETNAME(OBJCLASS)

// BaseLayer定义可以在外部new,这样可以外接插件只管处理逻辑
// layer知道自己的gpu类型.设计分为二种
// 其一是本身插件提供公共处理,配合factory.
// 其二是外部插件提供new obj,需要自己设定gpu类型
class ACOE_EXPORT BaseLayer : public IBaseLayer {
   protected:
    struct NodeIndex {
        int32_t nodeIndex = -1;
        int32_t siteIndex = -1;
    };

   protected:
    friend class PipeNode;
    friend class PipeGraph;
    friend class InputLayer;
    friend class GroupLayer;
    friend class BaseLayer;
    // ITLayer类的所有实例化, 都为BaseLayer的友元
    template <typename T>
    friend class ITLayer;

    GpuType gpu = GpuType::other;
    // 定义当前层需要的输入数量
    int32_t inCount = 1;
    // 定义当前层需要的输出数量
    int32_t outCount = 1;
    class PipeGraph* pipeGraph = nullptr;
    // weak_ptr用于检测是否sharp_ptr是否已经没有引用,避免指向野指针
    std::weak_ptr<class PipeNode> pipeNode;
    // 每个层的imagetype对应shader里类型,连接时需要检测
    std::vector<ImageFormat> inFormats;
    // 每个层的imagetype对应shader里类型,连接时需要检测
    std::vector<ImageFormat> outFormats;
    // 输入层
    bool bInput = false;
    // 输出层
    bool bOutput = false;
    // 是否自动适配上层ImageType
    bool bAutoImageType = false;
    // 自身是否不包含运算
    bool bNoCompute = false;

    // 每个输入节点对应一个输入
    std::vector<NodeIndex> inLayers;
    // 每个输出节点可以对应多个输出
    std::vector<std::vector<NodeIndex>> outLayers;

    std::string markStr = "";

   public:
    BaseLayer(/* args */) : BaseLayer(1, 1){};
    BaseLayer(int32_t inSize, int32_t outSize);
    virtual ~BaseLayer();

   public:
    // 附加到那个图表上
    class PipeGraph* getGraph();
    int32_t getInCount();
    int32_t getOutCount();

    // 如下所有公共方法全是转接PipeNode,需要附加到PipeGraph后才可以调用
   public:
    virtual bool bAttachGraph() final;
    virtual void setVisable(bool bvisable) final;
    virtual void setEnable(bool benable) final;
    virtual int32_t getGraphIndex() final;
    // 如果层有多个输入,可能不同输入对应不同层内不同层
    // index表示输入节点索引,node表示层内层节点,toInIndex表示对应层内层输入位置
    virtual void setStartNode(IBaseLayer* node, int32_t index = 0,
                              int32_t toInIndex = 0) final;
    virtual void setEndNode(IBaseLayer* node) final;
    virtual IBaseLayer* addNode(IBaseLayer* layer) final;
    virtual IBaseLayer* addNode(class ILayer* layer) final;
    virtual IBaseLayer* addLine(IBaseLayer* to, int32_t formOut = 0,
                                int32_t toIn = 0) final;
    virtual const char* getMark() final;

   public:
    virtual const char* getName();

   protected:
    // 附加到图表上的节点
    std::shared_ptr<class PipeNode> getNode();
    void cheackAttachGraph();
    bool addInLayer(int32_t inIndex, int32_t nodeIndex, int32_t outputIndex);
    bool addOutLayer(int32_t outIndex, int32_t nodeIndex, int32_t inIndex);
    bool vaildInLayers();
    void initLayer();
    void resetGraph();
    bool getInFormat(ImageFormat& format, int32_t index = 0);

   protected:
    // 添加进pipeGraph时调用
    virtual void onInit();
    // 添加pipeGraph赋值节点后,一般用于组合多节点层
    virtual void onInitNode(){};
    // 已经添加进pipeGraph,pipeGraph把所有层连接起来,此时知道inputFormats的长宽
    // 并根据当前层的需求,设定对应outFormats,也就是下一层的inputFormats
    // 可分配线程组的大小了
    virtual void onInitLayer(){};
    // 根据inputFormats初始化buffer
    virtual void onInitBuffer(){};
    // 更新参数,子类会有updateParamet(T t)保存参数,等到运行前提交执行
    virtual void onUpdateParamet() override {};
    virtual bool onFrame() = 0;
};

// GroupLayer自身不处理任何运算,只是组合运算层
class ACOE_EXPORT GroupLayer : public BaseLayer {
   public:
    GroupLayer();
    virtual ~GroupLayer();

   protected:
    virtual void onInit() override{};
    // 实现层内子层的连接顺序
    virtual void onInitNode() override = 0;
    virtual bool onFrame() override { return true; };
};

}  // namespace aoce