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
    std::string paramFile = "net/pfld-sim.param";
    std::string binFile = "net/pfld-sim.bin";
#if defined(__ANDROID__)
    AAssetManager* assetManager = AoceManager::Get().getAppEnv().assetManager;
    assert(assetManager != nullptr);
    int ret = net->load_param(assetManager, paramFile.c_str());
    if (ret == 0) {
        ret = net->load_model(assetManager, binFile.c_str());
    }
#else
    std::string paramPath = getAocePath() + "/" + paramFile;
    std::string binPath = getAocePath() + "/" + binFile;
    int ret = net->load_param(paramPath.c_str());
    if (ret == 0) {
        ret = net->load_model(binPath.c_str());
    }
#endif
    bInitNet = false;
    if (ret == 0) {
        ncnnInLayer = static_cast<VkNcnnInCropLayer*>(ncnnLayer);
        drawLayer = drawlayer;
        NcnnInParamet paramet = {};
        paramet.scale = {1 / 255.f, 1 / 255.f, 1 / 255.f, 1.0f};
        paramet.outWidth = netFormet.width;
        paramet.outHeight = netFormet.height;
        ncnnInLayer->updateParamet(paramet);
        ncnnInLayer->setObserver(this, netFormet.imageType);
        bInitNet = true;
        return true;
    }
    return false;
}

void FaceKeypointDetector::onResult(VulkanBuffer* buffer,
                                    const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();
    // 得到当前面部区域
    FaceBox faceBox = {};
    ncnnInLayer->getInFaceBox(faceBox);

    ncnn::Extractor netEx = net->create_extractor();
    ncnn::Mat inMat(netFormet.width, netFormet.height, 3, buffer->getCpuData());
    netEx.input("input_1", inMat);
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