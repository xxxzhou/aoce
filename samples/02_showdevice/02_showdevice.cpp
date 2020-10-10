#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace aoce;
using namespace cv;

#define ENUMSTR(str) #str

static cv::Mat* show = nullptr;
static int index = 0;
static int formatIndex = 0;

static void onDrawFrame(VideoFrame frame) {
    std::cout << "time stamp:" << frame.timeStamp << std::endl;
}

int main() {
    ModuleManager::Get().regAndLoad("aoce_win_mf");
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
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
    // for (const auto& vf : formats) {
    //     std::cout << "index:" << vf.index << " width: " << vf.width
    //               << " hight:" << vf.height << " fps:" << vf.fps
    //               << " format:" << to_string(vf.videoType) << std::endl;
    // }
    if (formats.size() > 355) {
        formatIndex = 355;
    }
    video->setFormat(formatIndex);
    video->open();
    auto& selectFormat = video->getSelectFormat();
    video->setVideoFrameHandle(onDrawFrame);
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
    ModuleManager::Get().unloadModule("aoce_win_mf");
}