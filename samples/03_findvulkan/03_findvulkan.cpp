#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace aoce;
using namespace cv;

static cv::Mat* show = nullptr;
static int index = 0;
static int formatIndex = 0;
static PipeGraph* vkGraph;
static InputLayer* inputLayer;
static OutputLayer* outputLayer;

static GpuType gpuType = GpuType::vulkan;

static void onDrawFrame(VideoFrame frame) {
    // std::cout << "time stamp:" << frame.timeStamp << std::endl;
    inputLayer->inputCpuData(frame.data[0]);
    vkGraph->run();
}

static void onImageProcessHandle(uint8_t* data, int32_t width, int32_t height,
                                 int32_t outIndex) {
    std::cout << "data:" << data[10000] << std::endl;
}

int main() {
    loadAoce();
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    // 生成一张执行图
    vkGraph = AoceManager::Get().getPipeGraphFactory(gpuType)->createGraph();
    auto* layerFactory = AoceManager::Get().getLayerFactory(gpuType);
    inputLayer = layerFactory->crateInput();
    outputLayer = layerFactory->createOutput();
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(outputLayer);
    // 设定输出函数回调
    outputLayer->setImageProcessHandle(onImageProcessHandle);
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    if (deviceList.size() > 1) {
        index = 1;
    }
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
    if (formats.size() > 355) {
        formatIndex = 355;
    }
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    video->setVideoFrameHandle(onDrawFrame);
    // 设定输入格式
    inputLayer->setImage(selectFormat);
    show = new cv::Mat(selectFormat.height, selectFormat.width, CV_8UC4);
    while (int key = cv::waitKey(20)) {
        cv::imshow("a", *show);
        if (key == 'q') {
            break;
        } else if (key == 'c') {
            video->close();
        } else if (key == 'o') {
            video->open();
        }
    }
    unloadAoce();
}