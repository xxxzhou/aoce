#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "aoce_vulkan_extra/VkExtraExport.h"

using namespace aoce;
using namespace cv;

static cv::Mat* show = nullptr;
static cv::Mat* show2 = nullptr;
static int index = 0;
static int formatIndex = 0;
static IPipeGraph* vkGraph;
static IInputLayer* inputLayer;
static IOutputLayer* outputLayer;
static IYUV2RGBALayer* yuv2rgbLayer;
// box模糊
static ITLayer<KernelSizeParamet>* boxFilterLayer;
static ITLayer<ChromaKeyParamet>* chromKeyLayer;
static ITLayer<AdaptiveThresholdParamet>* adaptiveLayer = nullptr;
static IBaseLayer* luminance = nullptr;
static IPerlinNoiseLayer* noiseLayer = nullptr;

static GpuType gpuType = GpuType::vulkan;

class TestCameraObserver : public IVideoDeviceObserver {
    virtual void onVideoFrame(VideoFrame frame) override {
        // std::cout << "time stamp:" << frame.timeStamp << std::endl;
        inputLayer->inputCpuData(frame.data[0]);
        vkGraph->run();
    }
};

class OutputLayerObserver : public IOutputLayerObserver {
    virtual void onImageProcess(uint8_t* data, const ImageFormat& format,
                                int32_t outIndex) final {
        // std::cout << "data:" << (int)data[10000] << std::endl;
        // std::vector<float> vecf(width*height * 4);
        // memcpy(vecf.data(), data, width * height * elementSize);
        memcpy(
            show->ptr<char>(0), data,
            format.width * format.height * getImageTypeSize(format.imageType));
    }
};

int main() {
    loadAoce();
    // 打开摄像机
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    VideoDevicePtr video = deviceList[index];
    std::string name = video->getName();
    std::string id = video->getId();
    std::cout << "name: " << name << std::endl;
    std::cout << "id: " << id << std::endl;
    auto& formats = video->getFormats();
    std::wcout << "formats count: " << formats.size() << std::endl;
    for (const auto& vf : formats) {
        std::cout << "index:" << vf.index << " width: " << vf.width
                  << " hight:" << vf.height << " fps:" << vf.fps
                  << " format:" << getVideoType(vf.videoType) << std::endl;
    }
    formatIndex = video->findFormatIndex(1920, 1080);
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    TestCameraObserver cameraObserver = {};
    video->setObserver(&cameraObserver);

    // 生成一张执行图
    vkGraph = getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({10, 10});

    chromKeyLayer = createChromaKeyLayer();
    ChromaKeyParamet keyParamet = {};
    keyParamet.ambientScale = 0.1f;
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.ambientColor = {0.6f, 0.1f, 0.6f};
    keyParamet.alphaCutoffMin = 0.001f;
    chromKeyLayer->updateParamet(keyParamet);

    adaptiveLayer = createAdaptiveThresholdLayer();
    AdaptiveThresholdParamet adaParamet = {};
    adaParamet.boxSize = 4;
    adaptiveLayer->updateParamet(adaParamet);

    luminance = createLuminanceLayer();

    noiseLayer = createPerlinNoiseLayer();
    noiseLayer->setImageSize(1920, 1080);

    VideoType videoType = selectFormat.videoType;
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
    if (getYuvIndex(videoType) > 0) {
        yuv2rgbLayer->updateParamet({videoType});
    }
    // 生成图
    vkGraph->addNode(inputLayer)
        ->addNode(yuv2rgbLayer)
        ->addNode(chromKeyLayer)
        ->addNode(outputLayer);
    // 设定输入格式
    inputLayer->setImage(selectFormat);
    // vkGraph->addNode(noiseLayer)->addNode(outputLayer);
    // vkGraph->addNode(inputLayer)->addNode(outputLayer);
    // 设定输出函数回调
    OutputLayerObserver outputObserver = {};
    outputLayer->setObserver(&outputObserver);
    //显示
    show = new cv::Mat(selectFormat.height, selectFormat.width, CV_8UC4);
    show2 = new cv::Mat(selectFormat.height, selectFormat.width, CV_8UC4);
    while (int key = cv::waitKey(30)) {
        cv::cvtColor(*show, *show2, cv::COLOR_RGBA2BGRA);
        cv::imshow("a", *show2);
        if (key == 'q') {
            break;
        } else if (key == 'c') {
            video->close();
        } else if (key == 'o') {
            video->open();
        }
    }
    video.reset();
    unloadAoce();
}