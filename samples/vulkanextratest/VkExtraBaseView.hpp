#pragma once

#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <string>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

#include "aoce_vulkan_extra/VkExtraExport.h"

using namespace aoce;
using namespace aoce::vulkan;

class VkExtraBaseView : public IVideoDeviceObserver {
   private:
    int index = 0;
    int formatIndex = 0;
    IPipeGraph* vkGraph = nullptr;
    IInputLayer* inputLayer = nullptr;
    IOutputLayer* outputLayer = nullptr;
    IYUV2RGBALayer* yuv2rgbLayer = nullptr;
    ITransposeLayer* transposeLayer = nullptr;
    ITexOperateLayer* operateLayer = nullptr;
    IReSizeLayer* resizeLayer = nullptr;
    ILayer* extraLayer = nullptr;

    IBaseLayer* layerNode = nullptr;
    VideoDevicePtr video = nullptr;

    GpuType gpuType = GpuType::vulkan;

    std::unique_ptr<VulkanWindow> window = nullptr;

    bool bAutoIn = false;

    std::mutex mtx;

   public:
    VkExtraBaseView();
    ~VkExtraBaseView();

   private:
        void onPreCommand(uint32_t index);

   public:
    virtual void onVideoFrame(VideoFrame frame) override;
    void initGraph(ILayer* layer, void* hinst, IBaseLayer* nextLayer = nullptr);
    void initGraph(std::vector<IBaseLayer*> layers, void* hinst,
                   bool bAutoIn = false);

    void openDevice(int32_t id = 0);
    void closeDevice();

    inline IOutputLayer* getOutputLayer() { return outputLayer; }

    void enableLayer(bool bEnable);

#if WIN32
    void run();
#endif
};