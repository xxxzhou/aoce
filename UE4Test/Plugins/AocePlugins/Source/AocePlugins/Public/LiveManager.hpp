#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "CameraProcess.hpp"
#include "VideoDisplay.hpp"
#include "VideoView.hpp"
#include "aoce/AoceCore.h"

namespace aoce {

	using namespace talkto;

	enum class RoleType : int32_t {
		other = 0,
		teacher,
		student,
		// 此项请保持最后
		max,
	};

	struct RoleSetting {
		// 对应Role的用户Id
		int32_t userId = 0;
		// 推流数
		int32_t pushCount = 1;
		// 拉流设置
		PullSetting pullSetting = {};
		// 推流设置
		std::vector<PushSetting> pushSettings;
	};

	class LiveManager : public ILiveObserver, public IOutputLayerObserver {
	private:
		/* data */
		LiveManager(/* args */);
		static LiveManager *instance;

	private:
		GpuType gpuType = GpuType::other;
		ILiveRoom *live = nullptr;
		std::string roomId = "";
		// 处理摄像机图像处理
		std::unique_ptr<CameraProcess> cameraProcess = nullptr;
		// 处理拉流图像处理,如果需求有多个,请声明多个处理
		std::unique_ptr<VideoView> videoView = nullptr;
		// 处理拉流图像显示处理
		std::unique_ptr<VideoDisplay> videoDisplay = nullptr;

		std::vector<RoleSetting> roleSettings;
		// 当前正在使用的角色
		RoleSetting roleSetting = {};

	public:
		static LiveManager &Get();
		~LiveManager();

	public:
		// 初始化直播模块
		void initLive(LiveType liveType, GpuType gpuType = GpuType::other);
		// 请确保成功初始化再调用
		ILiveRoom *getLive();
	private:
		int32_t getUserId(RoleType roleType);
		bool getRole(const int32_t& userId, RoleType &role, int32_t &userIndex);
	public:
		// 用于测试
		inline CameraProcess *getCameraProcess() { return cameraProcess.get(); }
		inline VideoDisplay* getVideoDisplay() { return videoDisplay.get(); }
	public:
		// formatIndex = -1,自动查找1080P分辨率
		void setCameraId(const char *mainCameraId, int32_t formatIndex = -1);
		// 一个角色可能有多个用户,比如多个学生,userIndex对应同一角色的用户编号
		void loginRoom(const char *roomId, RoleType role, int32_t userIndex = 0);
		void startPush();
		void closePush();
		void startPull(RoleType role, int32_t userIndex = 0);
		void closePull(RoleType role, int32_t userIndex = 0);
		void logoutRoom();
		void shutdownRoom();

	public:
		virtual void onImageProcess(uint8_t *data, const ImageFormat &format,
			int32_t outIndex) override;

	public:
		virtual void onEvent(int32_t operater, int32_t code, LogLevel level,
			const char *msg) override;

		virtual void onInitRoom() override;

		// loginRoom的网络应答
		virtual void onLoginRoom(bool bReConnect = false) override;

		// 加入的房间人数变化
		virtual void onUserChange(int32_t userId, bool bAdd) override;

		// 自己pushStream/stopPushStream推流回调,code不为0应该是出错了
		virtual void onStreamUpdate(int32_t index, bool bAdd,
			int32_t code) override;

		// 别的用户pullStream/stopPullStream拉流回调
		virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd,
			int32_t code) override;

		// 用户对应流的视频桢数据
		virtual void onVideoFrame(int32_t userId, int32_t index,
			const VideoFrame &videoFrame) override;

		// 用户对应流的音频桢数据
		virtual void onAudioFrame(int32_t userId, int32_t index,
			const AudioFrame &audioFrame) override;

		// 推流的质量
		virtual void onPushQuality(int32_t index, int32_t quality, float fps,
			float kbs) override;

		// 拉流质量
		virtual void onPullQuality(int32_t userId, int32_t index, int32_t quality,
			float fps, float kbs) override;

		// 拿出房间
		virtual void onLogoutRoom() override;
	};
}  // namespace aoce
