
#include <AoceManager.hpp>
#include <media/MediaPlayer.hpp>
#include <module/ModuleManager.hpp>
#include <thread>

#include "aoce/fixgraph/VideoView.hpp"
#include "aoce_win/DX11/Dx11Window.hpp"

// 是用摄像机还是用ffmpeg拉流
#define USE_CAMERA 1

using namespace aoce;
using namespace aoce::win;

#if USE_CAMERA
static int index = 0;
static int formatIndex = 0;  // 29;
#else
static HINSTANCE instance = nullptr;
static IMediaPlayer *player = nullptr;
static class TestMediaPlay *testPlay = nullptr;
static std::string uri = "rtmp://58.200.131.2:1935/livetv/hunantv";
#endif

static std::unique_ptr<Dx11Window> window = nullptr;
static std::unique_ptr<VideoView> viewGraph = nullptr;

// cuda/vulkan 分别对接不同的DX11输出
static GpuType gpuType = GpuType::cuda;

static void *dx11Device = nullptr;
static void *dx11Tex = nullptr;

static void onTick(void *dx11, void *tex) {
    std::string msg;
    string_format(msg, "time stamp: ", getNowTimeStamp());
    logMessage(LogLevel::info, msg);
    dx11Device = dx11;
    dx11Tex = tex;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    viewGraph->getOutputLayer()->outDx11GpuTex(dx11, dx11Tex);
}

#if USE_CAMERA
class TestCameraObserver : public IVideoDeviceObserver {
    virtual void onVideoFrame(VideoFrame frame) override {
        // std::cout << "time stamp:" << frame.timeStamp << std::endl;
        viewGraph->runFrame(frame);
        // viewGraph->getOutputLayer()->outDx11GpuTex(dx11Device, dx11Tex);
    }
};
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
    TestCameraObserver cameraObserver = {};
    VideoDevicePtr video = deviceList[index];
    video->setObserver(&cameraObserver);
    std::string name = video->getName();
    std::string id = video->getId();
    std::cout << "name: " << name << std::endl;
    std::cout << "id: " << id << std::endl;
    auto &formats = video->getFormats();
    std::wcout << "formats count: " << formats.size() << std::endl;
    for (const auto &vf : formats) {
        std::cout << "index:" << vf.index << " width: " << vf.width
                  << " hight:" << vf.height << " fps:" << vf.fps
                  << " format:" << getVideoType(vf.videoType) << std::endl;
    }
    formatIndex = video->findFormatIndex(1280, 720);
    video->setFormat(formatIndex);
    auto &selectFormat = video->getSelectFormat();
    VideoType videoType = selectFormat.videoType;
    // MF模式下,mjpg会转换成yuv2I
    if (selectFormat.videoType == VideoType::mjpg) {
        videoType = VideoType::yuv2I;
    }
#else
    instance = hInstance;
    player = AoceManager::Get()
                 .getMediaFactory(MediaType::ffmpeg)
                 ->createPlay();
    testPlay = new TestMediaPlay();
    player->setDataSource(uri.c_str());
    player->setObserver(testPlay);
    VideoType videoType = VideoType::yuv420P;
#endif

    viewGraph = std::make_unique<VideoView>(gpuType);
    viewGraph->getOutputLayer()->updateParamet({true, true});
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
