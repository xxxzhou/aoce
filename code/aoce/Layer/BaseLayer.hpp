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
    struct InputLayer {
        int32_t nodeIndex = -1;
        int32_t outputIndex = -1;
    };

   protected:
    friend class PipeNode;
    friend class PipeGraph;

    GpuType gpu = GpuType::other;
    // 定义当前层需要的输入数量
    int32_t inCount = 1;
    // 定义当前层需要的输出数量
    int32_t outCount = 1;
    class PipeGraph* pipeGraph = nullptr;
    // 每个层的imagetype对应shader里类型,连接时需要检测
    std::vector<ImageFormat> inFormats;
    // 每个层的imagetype对应shader里类型,连接时需要检测
    std::vector<ImageFormat> outFormats;
    bool bInput = false;
    bool bOutput = false;

    std::vector<InputLayer> inLayers;

   public:
    BaseLayer(/* args */) : BaseLayer(1, 1){};
    BaseLayer(int32_t inSize, int32_t outSize);
    virtual ~BaseLayer();

   public:
    virtual void onParametChange(){};
    class PipeGraph* getGraph();

   protected:
    bool addInLayer(int32_t inIndex, int32_t nodeIndex, int32_t outputIndex);
    bool vaildInLayers();
    void initLayer();

   protected:
    // 添加进pipeGraph
    virtual void onInit() = 0;
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

// 层不会单独从ILayer继承,还一个继承路径应该从BaseLayer来
class ILayer {
   public:
    // 请看上面宏AOCE_LAYER_QUERYINTERFACE提供的默认实现
    virtual BaseLayer* getLayer() = 0;
};

// 分离导致层不同参数的差异(AOCE_LAYER_QUERYINTERFACE)
template <typename T>
class ITLayer : public ILayer {
   protected:
    T oldParamet = {};
    T paramet = {};

   public:
    inline void updateParamet(const T& t) {
        oldParamet = this->paramet;
        this->paramet = t;
        getLayer()->onParametChange();
    };
};

// YUV 2 RGBA 转换
class YUV2RGBALayer : public ITLayer<YUVParamet> {};

// RGBA 2 YUV 转换
class RGBA2YUVLayer : public ITLayer<YUVParamet> {};

class TexOperateLayer : public ITLayer<TexOperateParamet> {};

// 纹理混合
struct BlendParamet {
    float right = 0.0f;
    float top = 0.0f;
    float width = 0.4f;
    float height = 0.4f;
    // 显示如上位置图像的透明度
    float alaph = 0.2f;
};

class BlendLayer : public ITLayer<BlendParamet> {};

}  // namespace aoce