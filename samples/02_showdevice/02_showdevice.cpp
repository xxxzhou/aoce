#include <Aoce.hpp>
#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace aoce;
using namespace cv;

#define ENUMSTR(str) #str

static cv::Mat* show = nullptr;
static int index = 0;
static int formatIndex = 0;

class TestCameraObserver : public IVideoDeviceObserver {
    virtual void onVideoFrame(VideoFrame frame) override {
        std::cout << "time stamp:" << frame.timeStamp << std::endl;
    }
};

int main() {
    ModuleManager::Get().regAndLoad("aoce_win_mf");
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    if (deviceList.size() > 1) {
        index = 1;
    }
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
    show = new cv::Mat(selectFormat.height, selectFormat.width, CV_8UC4);

    // std::string aocePath = getAocePath();
    // std::string imgPathI =
    //     aocePath + "/assets/images/lookup_amatorka.png";  // lookup_amatorka toy.bmp
    // std::string imgPathP = aocePath + "/assets/images/toy-mask.bmp";
    // cv::Mat I = cv::imread(imgPathI.c_str(), IMREAD_COLOR);
    // cv::cvtColor(I, I, cv::COLOR_BGR2RGB);
    // std::wstring spath =
    //     utf8TWstring(aocePath + "/assets/images/lookup_amatorka.binary");
    // saveFileBinary(spath.c_str(), (void*)I.data, 512 * 512 * 3);

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
