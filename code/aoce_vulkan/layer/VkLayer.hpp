#pragma once

#include <Layer/BaseLayer.hpp>

#include "../vulkan/VulkanContext.hpp"
#include "../vulkan/VulkanPipeline.hpp"
#include "../vulkan/VulkanShader.hpp"
#include "VkHelper.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

typedef std::shared_ptr<VulkanBuffer> VulkanBufferPtr;
typedef std::shared_ptr<VulkanTexture> VulkanTexturePtr;

// 如果CPU/GPU参数对应,并且不会导致graph重启
#define AOCE_VULKAN_PARAMETUPDATE()                                    \
   protected:                                                          \
    virtual inline void onUpdateParamet() override {                   \
        if (bParametMatch) {                                           \
            if (constBufCpu.size() == sizeof(paramet)) {               \
                memcpy(constBufCpu.data(), &paramet, sizeof(paramet)); \
            }                                                          \
            bParametChange = true;                                     \
        }                                                              \
    }

// 外部Vulkan层实现请继承这个类,提供对应计算图上的VulkanContext
class AOCE_VULKAN_EXPORT VkLayer : public BaseLayer {
    friend class VkPipeGraph;

   protected:
    // groupX/groupY 一定要与CS里的线程组划分对应,否则出现奇怪的问题
#if WIN32
    int32_t groupX = 16;
    int32_t groupY = 16;
#else
    int32_t groupX = 16;
    int32_t groupY = 16;
#endif
    // int32_t groupZ = 1;
    int32_t sizeX = 1;
    int32_t sizeY = 1;
    // 每个子类有必要的话,设定默认上传的第一个UBO大小,这个参数请在初始化时指定
    int32_t conBufSize = 0;
    // 一般来说,有二种,一种每桢运行前确定,一种是每桢运行时变化,有可能二种都有
    std::vector<uint8_t> constBufCpu;
    // 默认提供一个参数buf
    std::unique_ptr<VulkanBuffer> constBuf = nullptr;
    std::unique_ptr<UBOLayout> layout = nullptr;
    std::unique_ptr<VulkanShader> shader = nullptr;

    class VkPipeGraph* vkPipeGraph = nullptr;
    VulkanContext* context = nullptr;
    VkPipeline computerPipeline = VK_NULL_HANDLE;
    std::vector<VulkanTexturePtr> inTexs;
    std::vector<VulkanTexturePtr> outTexs;

    VkCommandBuffer cmd;
    // 如果CPU与GPU参数对应
    bool bParametMatch = false;
    // 非线程更新告诉更新数据线程需要更新UBO
    // 保证一个PipeGraph里都在更新数据线程里更新
    bool bParametChange = false;

    std::string glslPath = "";

   public:
    VkLayer(/* args */);
    ~VkLayer() override;

   protected:
    // 初始化时请指定
    void setUBOSize(int size, bool bMatchParamet = false);
    void createOutTexs();

   public:
    void generateLayout();
    void updateUBO(void* data);
    void submitUBO();

   protected:
    // 只发生在附加PipeGraph时,特定VK一些实现,如果要特定实现onInitGraph
    virtual void onInit() final;
    // 每次PipeGraph调用Reset时发生,此时知道输入层
    virtual void onInitLayer() override;
    // VK自动根据输入初始化相应的信息,如果要修改,请在PipeGraph实现
    virtual void onInitBuffer() final;
    virtual bool onFrame() override;
    // 根据输入节点返回是否需要sampled
    virtual bool getSampled(int32_t inIndex) { return false; };
    virtual bool sampledNearest(int32_t inIndex) { return false; };

   protected:
    // vulkan层在onInit后,shader可以放这编译,如果参数影响shader编译,需要放入onInitPipe
    virtual void onInitGraph();
    // onInitBuffer后,onInitBuffer已经关联后上层的输出当做本层的输入
    virtual void onInitVkBuffer(){};
    // 根据上面shader/buffer,组建计算管线
    virtual void onInitPipe();
    // 每桢onFrame之前调用,与onFrame/执行commandbuffer在同一线程.
    // 可以用来根据标记更新一些Vulkan 资源
    virtual void onPreFrame();
    // CommandBuffer
    virtual void onPreCmd();
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce