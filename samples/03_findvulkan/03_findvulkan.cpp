#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace aoce;
using namespace cv;

static cv::Mat* show = nullptr;
static cv::Mat* show2 = nullptr;
static int index = 1;
static int formatIndex = 0;
static IPipeGraph* vkGraph;
static IInputLayer* inputLayer;
static IOutputLayer* outputLayer;
static IYUVLayer* yuv2rgbLayer;

// cuda也可测试
static GpuType gpuType = GpuType::vulkan;

class TestCameraObserver : public IVideoDeviceObserver {
    virtual void onVideoFrame(VideoFrame frame) override {
        // std::cout << "time stamp:" << frame.timeStamp << std::endl;
        inputLayer->inputCpuData(frame);
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
    inputLayer = layerFactory->createInput();
    outputLayer = layerFactory->createOutput();
    yuv2rgbLayer = layerFactory->createYUV2RGBA();

    VideoType videoType = selectFormat.videoType;
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
    if (getYuvIndex(videoType) > 0) {
        yuv2rgbLayer->updateParamet({videoType});
    }
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer)->addNode(outputLayer);
    // vkGraph->addNode(inputLayer)->addNode(outputLayer);
    // 设定输出函数回调
    OutputLayerObserver outputObserver = {};
    outputLayer->setObserver(&outputObserver);
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
            vkGraph->clear();
            vkGraph->addNode(inputLayer)
                ->addNode(yuv2rgbLayer)               
                ->addNode(outputLayer);
            video->open();
        } else if (key == 'p') {
            vkGraph->reset();
        }
    }
    video.reset();
    unloadAoce();
}