#pragma once

#include <memory>

#include "../VkExtraExport.hpp"
#include "VkBlendingModeLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

// 用于保存一桢图像
// 用的有向无循环图,所以不能有回环线,保存一桢然后使用构成回环
// 定义这层为bInput = true,告诉外面不需要自动连接别层输入,手动连接
class VkSaveFrameLayer : public VkLayer {
    AOCE_LAYER_GETNAME(VkSaveFrameLayer)
   private:
    // 是否自己使用ComputeShader复制
    bool bUserPipe = true;

   public:
    VkSaveFrameLayer(/* args */);
    ~VkSaveFrameLayer();

   public:
    // 手动设定需要保存的纹理,需要保存层在onInitLayer调用
    // 这个时候层知道了纹理大小,层在graph的位置,然后下一时间onPreCmd可以汇总信息
    void saveImageInfo(const ImageFormat& imageFormat, int32_t nodeIndex,
                       int32_t outNodeIndex);

   protected:
    // 比较特殊,在这个时间,可能还拿不到inTexs数据,交给onPreCmd
    virtual void onInitPipe() override{};
    virtual void onCommand() override;
};

class VkLowPassLayer : public VkDissolveBlendLayer {
    AOCE_LAYER_GETNAME(VkSaveFrameLayer)
   private:
    /* data */
    std::unique_ptr<VkSaveFrameLayer> saveLayer = nullptr;

   public:
    VkLowPassLayer(/* args */);
    ~VkLowPassLayer();

   protected:
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
    virtual void onInitLayer() override;
};

class VkHighPassLayer : public VkDifferenceBlendLayer, public ITLayer<float> {
    AOCE_LAYER_QUERYINTERFACE(VkHighPassLayer)
   private:
    std::unique_ptr<VkLowPassLayer> lowLayer = nullptr;

   public:
    VkHighPassLayer();
    ~VkHighPassLayer();

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce