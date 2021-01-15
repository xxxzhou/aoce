#include <AoceCore.h>

#include <AoceManager.hpp>
#include <Media/MediaPlayer.hpp>
#include <Module/ModuleManager.hpp>
#include <thread>

#include "../aoce_win/DX11/Dx11Window.hpp"

// 是用摄像机还是用ffmpeg拉流
#define USE_CAMERA 1

using namespace aoce;
using namespace aoce::win;

#if USE_CAMERA
static int index = 0;
static int formatIndex = 29;
#else
static HINSTANCE instance = nullptr;
static MediaPlayer *player = nullptr;
static class TestMediaPlay *testPlay = nullptr;
static std::string uri = "rtmp://58.200.131.2:1935/livetv/hunantv";
#endif

static std::unique_ptr<Dx11Window> window = nullptr;
static std::unique_ptr<VideoViewGraph> viewGraph = nullptr;

static GpuType gpuType = GpuType::cuda;

static void onTick(void *dx11, void *tex) {
    std::string msg;
    string_format(msg, "time stamp: ", getNowTimeStamp());
    logMessage(LogLevel::info, msg);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    viewGraph->getOutputLayer()->outDx11GpuTex(dx11, tex);
}

#if USE_CAMERA
static void onDrawFrame(VideoFrame frame) {
    // std::cout << "time stamp:" << frame.timeStamp << std::endl;
    viewGraph->runFrame(frame);
}
#else
class TestMediaPlay : public IMediaPlayerObserver {
   public:
    TestMediaPlay() {}

    virtual ~TestMediaPlay() override{};

   public:
    virtual void onPrepared() override {        
        logMessage(LogLevel::info, "start");
        player->start();
    }

    virtual void onError(PlayStatus staus, int32_t code,
                         std::string msg) override {
        std::string mssg;
        string_format(mssg, "status: ", (int32_t)staus, " msg: ", msg);
        logMessage(LogLevel::info, mssg);
    };

    virtual void onVideoFrame(const VideoFrame &frame) override {
        std::string msg;
        string_format(msg, "time stamp: ", frame.timeStamp);
        logMessage(LogLevel::info, msg);
        viewGraph->runFrame(frame);
    };

    virtual void onStop() override{};

    virtual void onComplate() override{};
};
#endif

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    loadAoce();
    window = std::make_unique<Dx11Window>();

#if USE_CAMERA
    auto &deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << "deivce count:" << deviceList.size() << std::endl;
    VideoDevicePtr video = deviceList[index];
    video->setVideoFrameHandle(onDrawFrame);
    std::wstring name((wchar_t *)video->getName().data());
    std::wstring id((wchar_t *)video->getId().data());
    std::wcout << "name: " << name << std::endl;
    std::wcout << "id: " << id << std::endl;
    auto &formats = video->getFormats();
    std::wcout << "formats count: " << formats.size() << std::endl;
    for (const auto &vf : formats) {
        std::cout << "index:" << vf.index << " width: " << vf.width
                  << " hight:" << vf.height << " fps:" << vf.fps
                  << " format:" << to_string(vf.videoType) << std::endl;
    }
    if (formats.size() > 355) {
        formatIndex = 355;
    }
    video->setFormat(formatIndex);
    auto &selectFormat = video->getSelectFormat();
    VideoType videoType = selectFormat.videoType;
    // MF模式下,mjpg会转换成yuv2I
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
#else
    instance = hInstance;
    player = AoceManager::Get().getMediaPlayerFactory(MediaPlayType::ffmpeg)->createPlay();
    testPlay = new TestMediaPlay();
    player->setDataSource(uri.c_str());
    player->setObserver(testPlay);
    VideoType videoType = VideoType::yuv420P;
#endif

    viewGraph = std::make_unique<VideoViewGraph>();
    viewGraph->initGraph(gpuType);
    viewGraph->updateParamet({videoType}, {true, false}, {true, true});

#if USE_CAMERA
    // 打开摄像机
    video->open();
    window->initWindow(hInstance, selectFormat.width, selectFormat.height,
                       "dx11 test camera", onTick);
    window->run();
#else
    player->prepare(false);
    auto videoStream = player->getVideoStream();
    window->initWindow(instance, videoStream.width, videoStream.height,
                       "dx11 test ffmpeg", onTick);
    window->run();
#endif

    return 0;
}
