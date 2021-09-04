#pragma once

#include <memory>

#include "CNNHelper.hpp"
#include "VkNcnnInLayer.hpp"

namespace aoce {

#define NUM_FEATUREMAP 4
#define HARD_NMS 1
/* mix nms was been proposaled in paper blaze face, aims to minimize the
 * temporal jitter*/
#define BLENDING_NMS 2
#define MAX_FACE 5
// #define AOCE_ONLY_FACE 1

class FaceDetector : public virtual IFaceDetector,
                     public virtual INcnnInLayerObserver,
                     public virtual DrawProperty {
   private:
    /* data */
    float nms = 0.4f;
    float threshold = 0.6f;
    int anchorsNum = 0;
    std::unique_ptr<ncnn::Net> net = nullptr;
    // ncnn::Mat inMat;
    ImageFormat netFormet = {};

    std::vector<Box> anchors;
    IFaceObserver* observer = nullptr;
    std::vector<FaceBox> detectorFaces;

    FaceDetectorType detectorType = FaceDetectorType::face_landmark;

    VkNcnnInLayer* ncnnInLayer = nullptr;
    IDrawRectLayer* drawLayer = nullptr;
    INcnnInCropLayer* cropLayer = nullptr;

    DrawRectParamet drawRect = {};

    bool bInitNet = false;
    bool bVulkanInput = false;

   public:
    FaceDetector(/* args */);
    virtual ~FaceDetector();

   private:
    bool initNet(FaceDetectorType detectortype);
    void initAnchors();

   public:
    virtual void setObserver(IFaceObserver* observer) override;
    virtual void setFaceKeypointObserver(INcnnInCropLayer* cropLayer) override;
    virtual bool initNet(IBaseLayer* ncnnInLayer,
                         IDrawRectLayer* drawLayer) override;

   public:
    virtual void onResult(VulkanBuffer* buffer,
                          const ImageFormat& imageFormat) override;
};

}  // namespace aoce