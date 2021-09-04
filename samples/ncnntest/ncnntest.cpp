#include <aoce_ncnn/AoceNcnnExport.h>

#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <string>
#include <vulkan/VulkanContext.hpp>
#include <vulkan/VulkanWindow.hpp>

#if !VULKAN_SHOW
#include <opencv2/opencv.hpp>
using namespace cv;
#endif

using namespace aoce;
using namespace aoce::vulkan;

#if !VULKAN_SHOW
static cv::Mat* show = nullptr;
static cv::Mat* show2 = nullptr;
static cv::Rect rect = {};
static std::vector<cv::Point> cvPoints;
#endif

static int index = 0;
static int formatIndex = 0;
static IPipeGraph* vkGraph = nullptr;
static IInputLayer* inputLayer = nullptr;
static IOutputLayer* outputLayer = nullptr;
static IYUVLayer* yuv2rgbLayer = nullptr;
static IBaseLayer* ncnnLayer = nullptr;
static INcnnInCropLayer* ncnnPointLayer = nullptr;
static IDrawPointsLayer* drawPointsLayer = nullptr;
static IDrawRectLayer* drawRectLayer = nullptr;

static bool bNetLoad = false;
static bool bNetPointLoad = false;

static IFaceDetector* faceDetector = nullptr;
static class FaceObserver* faceObserver = nullptr;
static IFaceKeypointDetector* pointDetector = nullptr;
static class FaceKeypointObserver* pointObserver = nullptr;
static bool bFindFace = false;
static bool bFindPoint = false;

static std::unique_ptr<VulkanWindow> window = nullptr;

#define FACE_POINT_COUNT 98

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
#if !VULKAN_SHOW
        int32_t size =
            format.width * format.height * getImageTypeSize(format.imageType);
        memcpy(show->ptr<char>(0), data, size);
        if (bFindFace) {
            Scalar color(0, 255, 0);
            cv::rectangle(*show, rect, color);
            if (bFindPoint) {
                for (int i = 0; i < FACE_POINT_COUNT; i++) {
                    cv::circle(*show, cvPoints[i], 1, cv::Scalar(0, 0, 255));
                }
            }
        }
#endif
    }
};

class FaceObserver : public IFaceObserver {
    virtual void onDetectorBox(const FaceBox* boxs, int32_t lenght) final {
        std::cout << "count:" << lenght << std::endl;
        bFindFace = lenght > 0;
        for (size_t i = 0; i < lenght; i++) {
            std::cout << "index: " << i + 1 << " probability:" << boxs[i].s
                      << std::endl;
            // cv::Rect rect = {};
#if !VULKAN_SHOW
            rect.x = show->cols * boxs[i].x1;
            rect.y = show->rows * boxs[i].y1;
            rect.width = show->cols * (boxs[i].x2 - boxs[i].x1);
            rect.height = show->rows * (boxs[i].y2 - boxs[i].y1);
#endif
        }
    }
};

class FaceKeypointObserver : public IFaceKeypointObserver {
    virtual void onDetectorBox(const vec2* points, int32_t lenght) final {
        std::cout << "count:" << lenght << std::endl;
        bFindPoint = lenght > 0;
#if !VULKAN_SHOW
        for (size_t i = 0; i < std::min(lenght, FACE_POINT_COUNT); i++) {
            cvPoints[i].x = points[i].x * show->cols;
            cvPoints[i].y = points[i].y * show->rows;
        }
#endif
    }
};

void onPreCommand(uint32_t index) {
    VkImage winImage = window->images[index];
    VkCommandBuffer cmd = window->cmdBuffers[index];
    // 我们要把cs生成的图复制到正在渲染的图上,先改变渲染图的layout
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkOutGpuTex outTex = {};
    outTex.commandbuffer = cmd;
    outTex.image = winImage;
    outTex.width = window->width;
    outTex.height = window->height;
    outputLayer->outVkGpuTex(outTex);
    // 复制完成后,改变渲染图的layout准备呈现
    changeLayout(cmd, winImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}
#if VULKAN_SHOW
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
#else
int main() {
    cvPoints.resize(FACE_POINT_COUNT);
#endif
    loadAoce();

    ncnnLayer = createNcnnInLayer();
    ncnnPointLayer = createNcnnInCropLayer();

    drawPointsLayer = createDrawPointsLayer();
    drawRectLayer = createDrawRectLayer();

    // 面部识别
    faceObserver = new FaceObserver();
    faceDetector = createFaceDetector();
    faceDetector->setObserver(faceObserver);
    faceDetector->setFaceKeypointObserver(ncnnPointLayer);
    faceDetector->setDraw(1, vec4(0.0f, 1.0f, 0.0f, 1.0f));
    bNetLoad = faceDetector->initNet(ncnnLayer, drawRectLayer);
    // 面部关键点识别
    pointDetector = createFaceKeypointDetector();
    pointObserver = new FaceKeypointObserver();
    pointDetector->setObserver(pointObserver);
    bNetPointLoad = pointDetector->initNet(ncnnPointLayer, drawPointsLayer);
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
    formatIndex = video->findFormatIndex(1280, 720);
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
#if VULKAN_SHOW
    outputLayer->updateParamet({false, true});
#else
    outputLayer->updateParamet({true, false});
#endif

    VideoType videoType = selectFormat.videoType;
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
    if (getYuvIndex(videoType) > 0) {
        yuv2rgbLayer->updateParamet({videoType});
    }
    // 生成图
    vkGraph->addNode(inputLayer)->addNode(yuv2rgbLayer);
    vkGraph->addNode(ncnnLayer);
    vkGraph->addNode(ncnnPointLayer);
    vkGraph->addNode(drawPointsLayer);
    vkGraph->addNode(outputLayer);
    vkGraph->addNode(drawRectLayer);
    yuv2rgbLayer->getLayer()->addLine(ncnnLayer);
    yuv2rgbLayer->getLayer()->addLine(ncnnPointLayer->getLayer());
    yuv2rgbLayer->getLayer()
        ->addLine(drawRectLayer->getLayer())
        ->addLine(drawPointsLayer->getLayer())
        ->addLine(outputLayer->getLayer());
    // 设定输出函数回调
    OutputLayerObserver outputObserver = {};
    outputLayer->setObserver(&outputObserver);
    // 设定输入格式
    inputLayer->setImage(selectFormat);

#if VULKAN_SHOW
    window = std::make_unique<VulkanWindow>(onPreCommand, false);
    window->initWindow(hInstance, selectFormat.width, selectFormat.height,
                       "ncnn test");
    window->run();
#else
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
#endif
    video.reset();
    unloadAoce();
}