#pragma once
#include "BaseLayer.hpp"

namespace aoce {

struct InputParamet {
    int32_t bCpu = true;
    int32_t bGpu = false;
};

// inputlayer 应该从VkBaseLayer/Dx11BaseLayer/CudaBaseLayer继承
class ACOE_EXPORT InputLayer : public ITLayer<InputParamet> {
   protected:
    VideoFormat videoFormat = {};
    uint8_t* frameData = nullptr;
    std::vector<uint8_t> videoFrameData;

   protected:
    virtual void onSetImage(VideoFormat videoFormat, int32_t index = 0) = 0;
    virtual void onInputCpuData(uint8_t* data, int32_t index = 0){};
    virtual void onInputCpuData(const VideoFrame& videoFrame,
                                int32_t index = 0){};

   public:
    void setImage(VideoFormat videoFormat, int32_t index = 0);
    // 输入CPU数据,这个data需要与pipegraph同线程,因为从各方面考虑这个不会复制data里的数据.
    void inputCpuData(uint8_t* data, int32_t index = 0);
    void inputCpuData(const VideoFrame& videoFrame, int32_t index = 0);
};

}  // namespace aoce