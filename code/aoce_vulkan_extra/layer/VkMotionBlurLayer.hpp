#pragma once

#include <memory>

#include "../VkExtraExport.h"
#include "VkLowPassLayer.hpp"
#include "VkReduceLayer.hpp"
#include "aoce_vulkan/layer/VkLayer.hpp"
#include "aoce_vulkan/layer/VkOutputLayer.hpp"

namespace aoce {
namespace vulkan {
namespace layer {

class VkMotionBlurLayer : public VkLayer, public ITLayer<MotionBlurParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkMotionBlurLayer)
   private:
    /* data */
   public:
    VkMotionBlurLayer(/* args */);
    virtual ~VkMotionBlurLayer();

   private:
    void transformParamet();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
    virtual void onUpdateParamet() override;
    virtual void onInitLayer() override;
};

class VkZoomBlurLayer : public VkLayer, public ITLayer<ZoomBlurParamet> {
    AOCE_LAYER_QUERYINTERFACE(VkZoomBlurLayer)
    AOCE_VULKAN_PARAMETUPDATE()
   private:
    /* data */
   public:
    VkZoomBlurLayer(/* args */);
    virtual ~VkZoomBlurLayer();

   protected:
    virtual bool getSampled(int32_t inIndex) override;
};

class VkMotionDetectorLayer : public VkLayer,
                              public IMotionDetectorLayer,
                              public IOutputLayerObserver {
    AOCE_LAYER_QUERYINTERFACE(VkMotionDetectorLayer)
   private:
    std::unique_ptr<VkLowPassLayer> lowLayer = nullptr;
    std::unique_ptr<VkReduceLayer> avageLayer = nullptr;
    std::unique_ptr<VkOutputLayer> outLayer = nullptr;
    IMotionDetectorObserver* observer = nullptr;

   public:
    VkMotionDetectorLayer();
    virtual ~VkMotionDetectorLayer();

   public:
    virtual void setObserver(IMotionDetectorObserver* observer) final;

   public:
    virtual void onImageProcess(uint8_t* data, const ImageFormat& format,
                                int32_t outIndex) final;
    virtual void onFormatChanged(const ImageFormat &imageFormat,
                         int32_t outIndex) override {};
   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

}  // namespace layer
}  // namespace vulkan
}  // namespace aoce