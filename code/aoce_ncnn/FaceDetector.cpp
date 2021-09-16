#include "FaceDetector.hpp"

#include <AoceManager.hpp>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace aoce {

FaceDetector::FaceDetector(/* args */) {
    net = std::make_unique<ncnn::Net>();
    net->opt.use_vulkan_compute = true;
// net->opt.use_fp16_storage = false;
// net->opt.use_fp16_arithmetic = false;
// net->opt.use_fp16_packed = false;
// 网络输入图像格式
#if WIN32
    netFormet.width = 320;
    netFormet.height = 240;
#elif __ANDROID__
    netFormet.width = 160;
    netFormet.height = 120;
#endif
    netFormet.imageType = ImageType::bgr8;
    detectorFaces.resize(MAX_FACE);
}

FaceDetector::~FaceDetector() {}

void FaceDetector::initAnchors() {
    anchors.clear();
    std::vector<std::vector<float>> minBoxes = {{10.0f, 16.0f, 24.0f},
                                                {32.0f, 48.0f},
                                                {64.0f, 96.0f},
                                                {128.0f, 192.0f, 256.0f}};
    std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<int> whlist = {netFormet.width, netFormet.height};
    std::vector<std::vector<float>> featuremapSize;
    for (auto size : whlist) {
        std::vector<float> fmItem;
        for (float stride : strides) {
            fmItem.push_back(ceil(size / stride));
        }
        featuremapSize.push_back(fmItem);
    }
    /* generate prior anchors */
    for (int index = 0; index < NUM_FEATUREMAP; index++) {
        for (int j = 0; j < featuremapSize[1][index]; j++) {
            for (int i = 0; i < featuremapSize[0][index]; i++) {
                for (float k : minBoxes[index]) {
                    float xCenter =
                        (i + 0.5f) * strides[index] / netFormet.width;
                    float yCenter =
                        (j + 0.5f) * strides[index] / netFormet.height;
                    float w = k / netFormet.width;
                    float h = k / netFormet.height;
                    anchors.push_back({clip(xCenter, 1.0f), clip(yCenter, 1),
                                       clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    anchorsNum = anchors.size();
}

void FaceDetector::setObserver(IFaceObserver* observer) {
    this->observer = observer;
}

void FaceDetector::setFaceKeypointObserver(INcnnInCropLayer* cropLayer) {
    this->cropLayer = cropLayer;
}

bool FaceDetector::initNet(FaceDetectorType detectortype) {
    detectorType = detectortype;

    int32_t gpu_count = ncnn::get_gpu_count();
    net->set_vulkan_device(0);

    std::string paramFile = "net/slim_320.param";
    std::string binFile = "net/slim_320.bin";
    if (detectorType == FaceDetectorType::face_landmark) {
        paramFile = "net/face.param";
        binFile = "net/face.bin";
    }
    int32_t ret = loadNet(net.get(), paramFile, binFile);
    if (ret == 0) {
        initAnchors();
        return true;
    }
    return false;
}

bool FaceDetector::initNet(IBaseLayer* ncnnLayer, IDrawRectLayer* drawlayer) {
    if (bInitNet) {
        return true;
    }
    bool bInit = initNet(detectorType);
    ncnnInLayer = static_cast<VkNcnnInLayer*>(ncnnLayer);
    drawLayer = drawlayer;
    if (bInit && ncnnInLayer) {
        // const float meanVal[] = {104.f, 117.f, 123.f};
        NcnnInParamet paramet = {};
        paramet.mean = {104.f, 117.f, 123.f, 0.0f};
        paramet.outWidth = netFormet.width;
        paramet.outHeight = netFormet.height;
        ncnnInLayer->updateParamet(paramet, net->opt.use_fp16_storage);
        ncnnInLayer->setObserver(this, netFormet.imageType);
        if (!ncnnInLayer) {
            logMessage(
                LogLevel::warn,
                "FaceDetector initialize the network parameters ncnnlayer, "
                "please use the function createNcnnInLayer to create");
            return false;
        }        
        bInitNet = true;
        bVulkanInput = false;
    }
    return bInitNet;
}

void FaceDetector::onResult(ncnn::VkMat& vkMat,
                            const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();

    ncnn::Extractor netEx = net->create_extractor();
    ncnn::Mat boxMat, scoreMat, landmarkMat;

    netEx.input(0, vkMat);
    if (detectorType == FaceDetectorType::face_landmark) {
        // loc
        netEx.extract("output0", boxMat);
        // class
        netEx.extract("530", scoreMat);
        // landmark
        netEx.extract("529", landmarkMat);
    } else {
        netEx.extract("scores", scoreMat);
        netEx.extract("boxes", boxMat);
    }
    std::vector<FaceBox> totalBoxs;

    float* boxes = boxMat.channel(0);
    float* scores = scoreMat.channel(0);
    float* landmarks = nullptr;
    if (detectorType == FaceDetectorType::face_landmark) {
        landmarks = landmarkMat.channel(0);
    }
    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchorsNum; i++) {
        if (*(scores + 1) > threshold) {
            Box locBox = {};
            FaceBox result = {};
            // center
            locBox.cx = anchors[i].cx + *boxes * 0.1f * anchors[i].sx;
            locBox.cy = anchors[i].cy + *(boxes + 1) * 0.1f * anchors[i].sy;
            // size
            locBox.sx = anchors[i].sx * exp(*(boxes + 2) * 0.2f);
            locBox.sy = anchors[i].sy * exp(*(boxes + 3) * 0.2f);
            // 扩大下size
            locBox.sx = locBox.sx * 1.2f;
            locBox.sy = locBox.sy * 1.2f;

            result.x1 = std::max(0.0f, (locBox.cx - locBox.sx / 2.0f));
            result.y1 = std::max(0.0f, (locBox.cy - locBox.sy / 2.0f));
            result.x2 = std::min(1.0f, (locBox.cx + locBox.sx / 2.0f));
            result.y2 = std::min(1.0f, (locBox.cy + locBox.sy / 2.0f));
            result.s = *(scores + 1);
            if (detectorType == FaceDetectorType::face_landmark) {
                // landmark
                for (int j = 0; j < 5; ++j) {
                    result.point[j].x =
                        anchors[i].cx +
                        *(landmarks + (j << 1)) * 0.1f * anchors[i].sx;
                    result.point[j].y =
                        anchors[i].cy +
                        *(landmarks + (j << 1) + 1) * 0.1f * anchors[i].sy;
                }
            }
            totalBoxs.push_back(result);
        }
        boxes += 4;
        scores += 2;
        if (detectorType == FaceDetectorType::face_landmark) {
            landmarks += 10;
        }
    }
    std::sort(totalBoxs.begin(), totalBoxs.end(),
              [](const FaceBox& a, const FaceBox& b) { return a.s > b.s; });
    auto totalSize = totalBoxs.size();
    std::vector<int> merged(totalSize, 0);
    int32_t faceIndex = 0;
    for (int i = 0; i < totalSize; i++) {
        if (merged[i]) {
            continue;
        }
        merged[i] = 1;
        float h0 = totalBoxs[i].y2 - totalBoxs[i].y1 + 1;
        float w0 = totalBoxs[i].x2 - totalBoxs[i].x1 + 1;
        float area0 = h0 * w0;
        for (int j = i + 1; j < totalSize; j++) {
            if (merged[j]) {
                continue;
            }
            float inner_x0 = std::max(totalBoxs[i].x1, totalBoxs[j].x1);
            float inner_y0 = std::max(totalBoxs[i].y1, totalBoxs[j].y1);
            float inner_x1 = std::min(totalBoxs[i].x2, totalBoxs[j].x2);
            float inner_y1 = std::min(totalBoxs[i].y2, totalBoxs[j].y2);

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0) {
                continue;
            }
            float inner_area = inner_h * inner_w;

            float h1 = totalBoxs[j].y2 - totalBoxs[j].y1 + 1;
            float w1 = totalBoxs[j].x2 - totalBoxs[j].x1 + 1;

            float area1 = h1 * w1;
            float score;
            score = inner_area / (area0 + area1 - inner_area);
            // 检查包含相交的二BOX的重合面积,大于设定nms,丢弃
            if (score > nms) {
                merged[j] = 1;
            }
        }
        detectorFaces[faceIndex++] = totalBoxs[i];
        if (faceIndex >= MAX_FACE) {
            break;
        }
    }
    if (drawLayer) {
        drawRect.color = color;
        drawRect.radius = std::max(1, radius / 2);
        if (faceIndex > 0) {
            drawRect.rect.x = detectorFaces[0].x1;
            drawRect.rect.y = detectorFaces[0].x2;
            drawRect.rect.z = detectorFaces[0].y1;
            drawRect.rect.w = detectorFaces[0].y2;
        } else {
            drawRect.color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        drawLayer->updateParamet(drawRect);
    }
    if (cropLayer) {
        cropLayer->detectFaceBox(detectorFaces.data(), faceIndex);
    }
    if (observer) {
        observer->onDetectorBox(detectorFaces.data(), faceIndex);
    }

    long long time2 = getNowTimeStamp();
    std::string tmsg;
    string_format(tmsg, "face time:", time2 - time1);
    logMessage(LogLevel::info, tmsg);
}
}  // namespace aoce