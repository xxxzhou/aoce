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
    ILayer* extraLayer = nullptr;

    GpuType gpuType = GpuType::vulkan;

    std::unique_ptr<VulkanWindow> window = nullptr;

   public:
    VkExtraBaseView();
    ~VkExtraBaseView();

   private: 
    void onFrame(VideoFrame frame);

    void onPreCommand(uint32_t index);

   public:
    void initGraph(ILayer* layer, void* hinst);

    void openDevice();

    inline OutputLayer* getOutputLayer() {return outputLayer;}

#if WIN32
    void run();
#endif
};