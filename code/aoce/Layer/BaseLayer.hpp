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

// 每个从继承ILayer的类,请在类头文件里添加这个宏,或是自己实现
#define AOCE_LAYER_QUERYINTERFACE(OBJCLASS)           \
   public:                                            \
    virtual inline BaseLayer* getLayer() override {   \
        OBJCLASS* obj = static_cast<OBJCLASS*>(this); \
        return static_cast<BaseLayer*>(obj);          \
    }

// BaseLayer定义可以在外部new,这样可以外接插件只管处理逻辑
// layer知道自己的gpu类型.设计分为二种
// 其一是本身插件提供公共处理,配合factory.
// 其二是外部插件提供new obj,需要自己设定gpu类型
class ACOE_EXPORT BaseLayer {
   protected:
    struct NodeIndex {
        int32_t nodeIndex = -1;
        int32_t outputIndex = -1;
    };

   protected:
    friend class PipeNode;
    friend class PipeGraph;
    friend class InputLayer;
    friend class GroupLayer;
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
    // 是否自动拿上一层的ImageType
    bool bAutoImageType = false;

    std::vector<NodeIndex> inLayers;

   public:
    BaseLayer(/* args */) : BaseLayer(1, 1){};
    BaseLayer(int32_t inSize, int32_t outSize);
    virtual ~BaseLayer();

   public:
    // 附加到那个图表上
    class PipeGraph* getGraph();
    // 附加到图表上的节点
    std::shared_ptr<class PipeNode> getNode();

   protected:
    bool addInLayer(int32_t inIndex, int32_t nodeIndex, int32_t outputIndex);
    bool vaildInLayers();
    void initLayer();
    void resetGraph();

   public:
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
    virtual void onUpdateParamet(){};
    virtual bool onFrame() = 0;
};

// 实现层(非抽像层)不会单独从ILayer继承,还一个继承路径应该从BaseLayer来
class ACOE_EXPORT ILayer {
   public:
    // 请看上面宏AOCE_LAYER_QUERYINTERFACE提供的默认实现
    virtual BaseLayer* getLayer() = 0;

   public:
    class PipeNode* getLayerNode();
};

// 分离导致层不同参数的差异(AOCE_LAYER_QUERYINTERFACE)
template <typename T>
class ITLayer : public ILayer {
   protected:
    T oldParamet = {};
    T paramet = {};

   public:
    void updateParamet(const T& t) {
        oldParamet = this->paramet;
        this->paramet = t;
        getLayer()->onUpdateParamet();
    };

    T getParamet() { return paramet; }
};

// YUV 2 RGBA 转换
typedef ITLayer<YUVParamet> YUV2RGBALayer;
// RGBA 2 YUV 转换
typedef ITLayer<YUVParamet> RGBA2YUVLayer;
typedef ITLayer<TexOperateParamet> TexOperateLayer;
typedef ITLayer<TransposeParamet> TransposeLayer;
typedef ITLayer<ReSizeParamet> ReSizeLayer;
typedef ITLayer<BlendParamet> BlendLayer;
typedef ITLayer<YUVParamet> YUV2RGBALayer;

// // YUV 2 RGBA 转换
// class YUV2RGBALayer : public ITLayer<YUVParamet> {};
// // RGBA 2 YUV 转换
// class RGBA2YUVLayer : public ITLayer<YUVParamet> {};
// class TexOperateLayer : public ITLayer<TexOperateParamet> {};
// class TransposeLayer : public ITLayer<TransposeParamet> {};
// class ReSizeLayer : public ITLayer<ReSizeParamet> {};
// class BlendLayer : public ITLayer<BlendParamet> {};

}  // namespace aoce