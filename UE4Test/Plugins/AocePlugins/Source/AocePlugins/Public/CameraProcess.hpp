#pragma once

#include <iostream>
#include <string>

#include "VideoDisplay.hpp"
#include "VideoProcess.hpp"

namespace aoce {

	using namespace talkto;

	class CameraProcess {
	private:
		/* data */
		std::string cameraId = "";
		int32_t formatIndex = -1;
		// 处理当前选择的摄像机扣像显示相关图像处理
		std::unique_ptr<VideoProcess> videoProcess = nullptr;
		// 图像显示处理
		std::unique_ptr<VideoDisplay> videoDisplay = nullptr;
		// 当前处理的摄像机
		IVideoDevice* video = nullptr;
		// 当前图像格式
		VideoFormat videoFromat = {};

	public:
		CameraProcess(GpuType gpuType = GpuType::other);
		~CameraProcess();

	public:
		// 用于测试
		inline VideoProcess* getVideoProcess() { return videoProcess.get(); }
		inline VideoDisplay* getVideoDisplay() { return videoDisplay.get(); }
		IMattingLayer* getMattingLayer();

	public:
		bool setCamera(const char* deviceId, int32_t formatIndex = -1);

		VideoFormat getVideoFormat();

		// ProcessType控制输出(原图/扣像图/推流编码),IOutputLayerObserver推流编码接收者
		bool openCamera(ProcessType processType,
			IOutputLayerObserver* handle = nullptr);

		void closeCamera();

		void updateMatting(const MattingParamet& mp);
		void updateOperate(const TexOperateParamet& op);
	};

}  // namespace aoce