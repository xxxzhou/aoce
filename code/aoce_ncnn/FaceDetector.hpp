#pragma once

#include <memory>

#include "CNNHelper.hpp"

namespace aoce {

#define NUM_FEATUREMAP 4
#define HARD_NMS 1
/* mix nms was been proposaled in paper blaze face, aims to minimize the
 * temporal jitter*/
#define BLENDING_NMS 2
#define MAX_FACE 5
// #define AOCE_ONLY_FACE 1

class FaceDetector : public IFaceDetector {
   private:
    /* data */
    float nms = 0.4f;
    float threshold = 0.6f;
    int anchorsNum = 0;
    std::unique_ptr<ncnn::Net> net = nullptr;
    ImageFormat netFormet = {};

    std::vector<Box> anchors;
    IFaceDetectorObserver* observer = nullptr;
    std::vector<FaceBox> detectorFaces;

    FaceDetectorType detectorType = FaceDetectorType::face;

   public:
    FaceDetector(/* args */);
    virtual ~FaceDetector();
    void initAnchors();

   public:
    virtual void setObserver(IFaceDetectorObserver* observer) override;
    virtual bool initNet(FaceDetectorType detectorType) override;
    virtual void detect(uint8_t* data, const ImageFormat& inFormat) override;
};

}  // namespace aoce