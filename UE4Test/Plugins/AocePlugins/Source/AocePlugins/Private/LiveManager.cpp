#include "LiveManager.hpp"
using namespace std::placeholders;

namespace aoce {

	LiveManager *LiveManager::instance = nullptr;

	LiveManager &LiveManager::Get() {
		if (instance == nullptr) {
			instance = new LiveManager();
		}
		return *instance;
	}

	LiveManager::LiveManager(/* args */) {
		int32_t roleSize = (int32_t)RoleType::max;
		roleSettings.resize(roleSize);
		for (int32_t i = 0; i < roleSize; i++) {
			RoleType roleType = (RoleType)i;
			roleSettings[i].userId = getUserId(roleType);
			int32_t pushCount = 1;
			if (roleType == RoleType::other) {
				pushCount = 0;
			}
			roleSettings[i].pushCount = pushCount;
			roleSettings[i].pushSettings.resize(pushCount);
			// 设定老师推视频,学生不推,根据需求设定这里
			if (pushCount > 0) {
				roleSettings[i].pushSettings[0].bVideo =
					roleType == RoleType::teacher;
			}
		}
		videoDisplay = std::make_unique<VideoDisplay>();
	}

	LiveManager::~LiveManager() {}

	void LiveManager::initLive(LiveType liveType, GpuType gpuType) {
		this->gpuType = gpuType;
		// live在unload时会自动清理
		live = getLiveRoom(liveType);
		AgoraContext contex = {};
		contex.bLoopback = false;
		bool bInit = live->initRoom(&contex, this);
		// 根据业务处理
		if (!bInit) {
			logMessage(LogLevel::error, "init live model failed.");
		}
		// other会自动选择,如WIN32自动选择CUDA,android自动选择VULKAN
		cameraProcess = std::make_unique<CameraProcess>(gpuType);
		videoView = std::make_unique<VideoView>(gpuType);
		// videoDisplay输出
		videoDisplay->setOutLayer(videoView->getOutputLayer());
	}

	ILiveRoom *LiveManager::getLive() {
		assert(live != nullptr);
		return live;
	}

	// 用于与老版本兼容
	int32_t LiveManager::getUserId(RoleType roleType) {
		switch (roleType) {
		case RoleType::teacher:
			return 600;
		case RoleType::student:
			return 700;
		default:
			return 1000;
		}
		return 1000;
	}

	bool LiveManager::getRole(const int32_t& userId, RoleType &role, int32_t &userIndex) {
		userIndex = 0;
		bool result = true;
		role = RoleType::other;
		if (userId == 1000) {
			role = RoleType::other;
		}
		else if (userId == 600) {
			role = RoleType::teacher;
		}
		else if (userId >= 700 && userId < 800) {
			role = RoleType::student;
			userIndex = userId - 700;
		}
		else {
			result = false;
		}
		return result;
	}

	void LiveManager::setCameraId(const char *mainCameraId, int32_t formatIndex) {
		cameraProcess->setCamera(mainCameraId, formatIndex);
	}

	void LiveManager::loginRoom(const char *roomId, RoleType role,
		int32_t userIndex) {
		this->roomId = roomId;
		int userId = getUserId(role) + userIndex;
		roleSetting = roleSettings[(int32_t)role];
		// 自己的id
		roleSetting.userId = userId;
		live->loginRoom(roomId, userId, roleSetting.pushCount);
	}

	void LiveManager::onImageProcess(uint8_t *data, const ImageFormat &format,
		int32_t outIndex) {
		VideoFrame videoFrame = {};
		uint8_t *yuvdata = (uint8_t *)data;
		videoFrame.width = format.width;
		videoFrame.height = format.height / 2;
		videoFrame.data[0] = data;
		videoFrame.videoType = VideoType::yuy2P;
		live->pushVideoFrame(0, videoFrame);
	}

	void LiveManager::onEvent(int32_t operater, int32_t code, LogLevel level,
		const char *msg) {
		// 根据业务处理,其中error需要处理,warn需要记录
		// std::string str;
		// string_format(str, "code: ", code, " msg: ", msg);
		// logMessage((LogLevel)level, str.c_str());
	}

	void LiveManager::onInitRoom() {};

	// loginRoom的网络应答
	void LiveManager::onLoginRoom(bool bReConnect) { startPush(); }

	void LiveManager::startPush() {
		// 先只考虑推一个流的情况
		if (roleSetting.pushCount > 0) {
			// 需要打开摄像机
			if (roleSetting.pushSettings[0].bVideo) {
				VideoFormat vf = cameraProcess->getVideoFormat();
				VideoStream &videoStream = roleSetting.pushSettings[0].videoStream;
				videoStream.fps = vf.fps;
				videoStream.width = vf.width;
				videoStream.height = vf.height;
				videoStream.videoType = VideoType::yuy2P;
				// 推流(ProcessType::transport)
				// 推流并本地显示(ProcessType::transport | ProcessType::matting)
				bool bOpen = cameraProcess->openCamera(
					ProcessType::transport | ProcessType::matting, this);
				// 如果摄像机没有正确打开
				if (!bOpen) {
				}
			}
			bool bpush = live->pushStream(0, roleSetting.pushSettings[0]);
			if (!bpush) {
			}
		}
	}

	void LiveManager::closePush() {
		// 如果摄像机是打开的,先关闭
		cameraProcess->closeCamera();
		live->stopPushStream(0);
	}

	void LiveManager::startPull(RoleType role, int32_t userIndex) {
		int32_t userId = roleSettings[int32_t(role)].userId + userIndex;
		// std::string str;
		// string_format(str, "pull user : ", userId, " stream");
		// logMessage(LogLevel::info, str.c_str());
		PullSetting setting = {};
		bool bPull = live->pullStream(userId, 0, setting);
		if (!bPull) {
		}
	}

	void LiveManager::closePull(RoleType role, int32_t userIndex) {
		int32_t userId = roleSettings[int32_t(role)].userId + userIndex;
		// std::string str;
		// string_format(str, "stop pull user : ", userId, " stream");
		// logMessage(LogLevel::info, str.c_str());
		PullSetting setting = {};
		live->stopPullStream(userId, 0);
	}

	void LiveManager::logoutRoom() {
		if (live) {
			// 关闭推流
			closePush();
			// 登出,自动关闭所有拉流
			live->logoutRoom();
		}
	}

	void LiveManager::shutdownRoom() {
		if (live) {
			live->shutdownRoom();
		}
	}

	// 加入的房间人数变化
	void LiveManager::onUserChange(int32_t userId, bool bAdd) {
		RoleType role = RoleType::other;
		int32_t userIndex = 0;
		// 非设置的用户Id登录到这个房间了,请检查这个情况
		if (!getRole(userId, role, userIndex)) {
			logMessage(LogLevel::warn, "not setting role");
			return;
		}
		if (bAdd) {
			startPull(role, userIndex);
		}
		else {
			closePull(role, userIndex);
		}
	}

	// 自己pushStream/stopPushStream推流的直播服务器回调,code不为0应该是出错了
	// bAdd=true是推流,否则是关闭推流
	void LiveManager::onStreamUpdate(int32_t index, bool bAdd, int32_t code) {
		// 用于验证直播服务器是否正确推流
		if (code < 0) {
			logMessage(LogLevel::warn, "push stream error.");
		}
	};

	// 对应上面onUserChange里的pullStream/stopPullStream服务器回调
	void LiveManager::onStreamUpdate(int32_t userId, int32_t index, bool bAdd,
		int32_t code) {
		if (code < 0) {
			RoleType role = RoleType::other;
			int32_t userIndex = 0;
			// 非设置的用户Id登录到这个房间了,请检查这个情况
			if (!getRole(userId, role, userIndex)) {
				logMessage(LogLevel::warn, "not setting role");
			}
			// std::string str;
			// string_format(str, (bAdd ? "add" : "remove"), "user: ", userId,
			//               " failed");
			// logMessage(LogLevel::info, str.c_str());
		}
	}

	// 用户对应流的视频桢数据
	void LiveManager::onVideoFrame(int32_t userId, int32_t index,
		const VideoFrame &videoFrame) {
		videoView->runFrame(videoFrame,true);
	}

	// 用户对应流的音频桢数据
	void LiveManager::onAudioFrame(int32_t userId, int32_t index,
		const AudioFrame &audioFrame) {
		// 拉流的角色
		RoleType role = RoleType::other;
		// 角色对应的用户索引
		int32_t userIndex = 0;
		// 非设置的用户Id登录到这个房间了,请检查这个情况
		if (!getRole(userId, role, userIndex)) {
			logMessage(LogLevel::warn, "not setting role");
			return;
		}
	}

	// 推流的质量
	void LiveManager::onPushQuality(int32_t index, int32_t quality, float fps,
		float kbs) {};

	// 拉流质量
	void LiveManager::onPullQuality(int32_t userId, int32_t index, int32_t quality,
		float fps, float kbs) {};

	// 登出房间服务器回调
	void LiveManager::onLogoutRoom() {};

}  // namespace aoce
