#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "aoce_vulkan_extra/VkExtraExport.hpp"

using namespace aoce;
using namespace cv;
using namespace aoce::vulkan;

static cv::Mat* show = nullptr;
static cv::Mat* show2 = nullptr;
static int index = 0;
static int formatIndex = 0;
static PipeGraph* vkGraph;
static InputLayer* inputLayer;
static OutputLayer* outputLayer;
static YUV2RGBALayer* yuv2rgbLayer;
// box模糊
static ITLayer<KernelSizeParamet>* boxFilterLayer;
static ITLayer<ChromKeyParamet>* chromKeyLayer;
static ITLayer<AdaptiveThresholdParamet>* adaptiveLayer = nullptr;
static BaseLayer* luminance = nullptr;

static GpuType gpuType = GpuType::vulkan;

static void onDrawFrame(VideoFrame frame) {
    // std::cout << "time stamp:" << frame.timeStamp << std::endl;
    inputLayer->inputCpuData(frame.data[0]);
    vkGraph->run();
}

static void onImageProcessHandle(uint8_t* data, ImageFormat format,
                                 int32_t outIndex) {
    // std::cout << "data:" << (int)data[10000] << std::endl;
    // std::vector<float> vecf(width*height * 4);
    // memcpy(vecf.data(), data, width * height * elementSize);
    memcpy(show->ptr<char>(0), data,
           format.width * format.height * getImageTypeSize(format.imageType));
}

int main() {
    loadAoce();
    // 打开摄像机
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    VideoDevicePtr video = deviceList[index];
    std::wstring name((wchar_t*)video->getName().data());
    std::wstring id((wchar_t*)video->getId().data());
    std::wcout << "name: " << name << std::endl;
    std::wcout << "id: " << id << std::endl;
    auto& formats = video->getFormats();
    std::wcout << "formats count: " << formats.size() << std::endl;
    for (const auto& vf : formats) {
        std::cout << "index:" << vf.index << " width: " << vf.width
                  << " hight:" << vf.height << " fps:" << vf.fps
                  << " format:" << to_string(vf.videoType) << std::endl;
    }
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    video->setVideoFrameHandle(onDrawFrame);

    // 生成一张执行图
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    yuv2rgbLayer = layerFactory->createYUV2RGBA();
    boxFilterLayer = createBoxFilterLayer();
    boxFilterLayer->updateParamet({10, 10});

    chromKeyLayer = createChromKeyLayer();
    ChromKeyParamet keyParamet = {};
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
    // vkGraph->addNode(inputLayer)->addNode(outputLayer);
    // 设定输出函数回调
    outputLayer->setImageProcessHandle(onImageProcessHandle);
    // 设定输入格式
    inputLayer->setImage(selectFormat);
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