#pragma once

#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <string>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

#include "aoce_vulkan_extra/VkExtraExport.hpp"

using namespace aoce;
using namespace aoce::vulkan;

class VkExtraBaseView {
   private:
    int index = 0;
    int formatIndex = 0;
    PipeGraph* vkGraph = nullptr;
    InputLayer* inputLayer = nullptr;
    OutputLayer* outputLayer = nullptr;
    YUV2RGBALayer* yuv2rgbLayer = nullptr;
    TransposeLayer* transposeLayer = nullptr;
    TexOperateLayer* operateLayer = nullptr;
    ReSizeLayer* resizeLayer = nullptr;
    ILayer* extraLayer = nullptr;

    PipeNodePtr yuvNode = nullptr;
    // PipeNodePtr layerNode = nullptr;
    VideoDevicePtr video = nullptr;

    GpuType gpuType = GpuType::vulkan;

    std::unique_ptr<VulkanWindow> window = nullptr;

    std::mutex mtx;

   public:
    VkExtraBaseView();
    ~VkExtraBaseView();

   private:
    void onFrame(VideoFrame frame);

    void onPreCommand(uint32_t index);

   public:
    void initGraph(ILayer* layer, void* hinst, BaseLayer* nextLayer = nullptr);
    void initGraph(std::vector<BaseLayer*> layers,void* hinst);

    void openDevice(int32_t id = 0);
    void closeDevice();

    inline OutputLayer* getOutputLayer() { return outputLayer; }

    void enableLayer(bool bEnable);

#if WIN32
    void run();
#endif
};