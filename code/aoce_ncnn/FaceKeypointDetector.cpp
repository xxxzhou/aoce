#include "FaceKeypointDetector.hpp"

#include "aoce/AoceManager.hpp"

namespace aoce {

float expSpecial(float x) {
    int gate = 1;
    if (abs(x) < gate) {
        return x * exp(gate);
    }
    if (x > 0) {
        return exp(x);
    } else {
        return -exp(-x);
    }
}

FaceKeypointDetector::FaceKeypointDetector(/* args */) {
    net = std::make_unique<ncnn::Net>();
    net->opt.use_vulkan_compute = true;
// 网络输入图像格式
    netFormet.width = 112;
    netFormet.height = 112;

    netFormet.imageType = ImageType::bgr8;
}

FaceKeypointDetector::~FaceKeypointDetector() {}

void FaceKeypointDetector::setObserver(IFaceKeypointObserver* observer) {
    this->observer = observer;
}

bool FaceKeypointDetector::initNet(INcnnInCropLayer* ncnnLayer,
                                   IDrawPointsLayer* drawlayer) {
    // m3_pfld_1029 pfld-sim
    if (bInitNet) {
        return true;
    }
    std::string paramFile = "net/pfld-sim.param";
    std::string binFile = "net/pfld-sim.bin";
    int32_t ret = loadNet(net.get(), paramFile, binFile);
    if (ret == 0) {
        ncnnInLayer = static_cast<VkNcnnInCropLayer*>(ncnnLayer);
        if (!ncnnInLayer) {
            logMessage(
                LogLevel::warn,
                "FaceKeypointDetector initialize the network parameters "
                "ncnnlayer, "
                "please use the function createNcnnInCropLayer to create");
            return false;
        }
        drawLayer = drawlayer;
        NcnnInParamet paramet = {};
        paramet.scale = {1 / 255.f, 1 / 255.f, 1 / 255.f, 1.0f};
        paramet.outWidth = netFormet.width;
        paramet.outHeight = netFormet.height;
        ncnnInLayer->updateParamet(paramet, net->opt.use_fp16_storage);
        ncnnInLayer->setObserver(this, netFormet.imageType);
        bInitNet = true;
    }
    return bInitNet;
}

void FaceKeypointDetector::onResult(ncnn::VkMat& vkMat,
                                    const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();
    // 得到当前面部区域
    FaceBox faceBox = {};
    ncnnInLayer->getInFaceBox(faceBox);

    ncnn::Extractor netEx = net->create_extractor();
    netEx.input("input_1", vkMat);
    ncnn::Mat out;
    netEx.extract("415", out);

    int32_t num = out.w / 2;
    std::vector<vec2> keypoints(num);

    float width = faceBox.x2 - faceBox.x1;
    float height = faceBox.y2 - faceBox.y1;
    for (int j = 0; j < num; j++) {
        keypoints[j].x = faceBox.x1 + out[j * 2] * width;
        keypoints[j].y = faceBox.y1 + out[j * 2 + 1] * height;
    }
    if (drawLayer) {
        drawLayer->drawPoints(keypoints.data(), keypoints.size(), color,
                              radius);
    }
    if (observer) {
        observer->onDetectorBox(keypoints.data(), keypoints.size());
    }
    long long time2 = getNowTimeStamp();
    std::string tmsg;
    string_format(tmsg, "face point time:", time2 - time1);
    logMessage(LogLevel::info, tmsg);
}

}  // namespace aoce