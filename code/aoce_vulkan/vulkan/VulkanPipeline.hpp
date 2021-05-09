#pragma once
// 从渲染来说,管线大致有几种,透明物体/非透明物体/UI/后处理特效,纯计算
// 管线需要设置的值大致对应shader相关
// 从UBO,Texture
#include <map>
#include <string>

#include "VulkanCommon.hpp"
namespace aoce {
namespace vulkan {

#if _WIN32
template class AOCE_VULKAN_EXPORT std::vector<VkDynamicState>;
template class AOCE_VULKAN_EXPORT std::vector<VkDescriptorSetLayout>;
template class AOCE_VULKAN_EXPORT std::vector<int32_t>;
template class AOCE_VULKAN_EXPORT std::map<VkDescriptorType, uint32_t>;
template class AOCE_VULKAN_EXPORT std::vector<std::vector<VkDescriptorSet>>;
#endif

// 可以由VulkanPipeline创建一个默认填充的FixPipelineState.
// 根据渲染非透明/透明/后处理/GUBFFER/阴影 不同条件修改FixPipelineState
struct FixPipelineState {
   public:
    // ---图元装配
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
    // ---光栅化
    VkPipelineRasterizationStateCreateInfo rasterizationState = {};
    // ---片版与老的桢缓冲像素混合
    VkPipelineColorBlendAttachmentState coolrAttach = {};
    VkPipelineColorBlendStateCreateInfo colorBlendState = {};
    // ---深度和模板测试
    VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
    // ---视口和裁剪矩形
    VkPipelineViewportStateCreateInfo viewportState = {};
    // ---多重采集(可以在渲染GBUFFER关闭,渲染最终颜色时打开)
    VkPipelineMultisampleStateCreateInfo multisampleState = {};
    // ---管线动态修改
    std::vector<VkDynamicState> dynamicStateEnables = {};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
};

struct UBOLayoutItem {
    VkDescriptorType descriptorType;
    // 可以组合,简单来说,UBO可以绑定到几个shader阶段
    VkShaderStageFlags shaderStageFlags;
};
#if _WIN32
template class AOCE_VULKAN_EXPORT std::vector<std::vector<UBOLayoutItem>>;
#endif
// 对应shader里固定结构的结构以及数据
class AOCE_VULKAN_EXPORT UBOLayout {
   public:
    UBOLayout();
    ~UBOLayout();

   private:
    VkDevice device = VK_NULL_HANDLE;
    // std::vector<UBOLayoutItem> items;
    std::vector<std::vector<UBOLayoutItem>> items;
    std::vector<int32_t> groupSize;
    std::map<VkDescriptorType, uint32_t> descripts;
    // 确定所有UBO不同descriptorType的总结
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    // 根据groupIndex分组生成不同VkDescriptorSetLayout
    std::vector<VkDescriptorSetLayout> descSetLayouts;

   public:
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    // 根据layout生成不同的
    std::vector<std::vector<VkDescriptorSet>> descSets;

   public:
    // 一个layout表示一个VkDescriptorSetLayout,shader上的一个set,layout里面索引对应binding
    // count表示一个group有几个set,主要用于一个UBO结构填充多处不同内容
    int32_t addSetLayout(std::vector<UBOLayoutItem>& layout,
                         uint32_t count = 1);
    void generateLayout(int32_t constSize = 0);

    void updateSetLayout(uint32_t groupIndex, uint32_t setIndex, ...);

    void updateSetLayout(uint32_t groupIndex, uint32_t setIndex,
                         std::vector<void*> bufferInfos);
};

class AOCE_VULKAN_EXPORT VulkanPipeline {
   private:
    /* data */
   public:
    VulkanPipeline(/* args */);
    ~VulkanPipeline();

   public:
    // 创建一个默认状态的管线
    static void createDefaultFixPipelineState(FixPipelineState& fixState);

    static VkPipelineShaderStageCreateInfo loadShader(
#if __ANDROID__
        AAssetManager* assetManager,
#endif
        VkDevice device, const std::string& fileName,
        VkShaderStageFlagBits stage);

    static VkComputePipelineCreateInfo createComputePipelineInfo(
        VkPipelineLayout layout, VkPipelineShaderStageCreateInfo stageInfo);
};
}  // namespace vulkan
}  // namespace aoce