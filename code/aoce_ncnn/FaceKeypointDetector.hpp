#pragma once

#include "CNNHelper.hpp"
#include "VkNcnnInLayer.hpp"
#include "net.h"

namespace aoce {

class FaceKeypointDetector : public virtual IFaceKeypointDetector,
                             public virtual INcnnInLayerObserver,
                             public virtual DrawProperty {
   private:
    /* data */
    std::unique_ptr<ncnn::Net> net = nullptr;
    ImageFormat netFormet = {};
    IFaceKeypointObserver* observer = nullptr;
    VkNcnnInCropLayer* ncnnInLayer = nullptr;
    IDrawPointsLayer* drawLayer = nullptr;
    bool bInitNet = false;

   public:
    FaceKeypointDetector(/* args */);
    ~FaceKeypointDetector();

   public:
    virtual void setObserver(IFaceKeypointObserver* observer) override;
    virtual bool initNet(INcnnInCropLayer* ncnnInLayer,
                         IDrawPointsLayer* drawLayer) override;

   public:
    virtual void onResult(VulkanBuffer* buffer,
                          const ImageFormat& imageFormat) override;
};

}  // namespace aoce