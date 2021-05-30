#include "CameraProcess.hpp"

#include "CameraManager.hpp"

namespace aoce {

	CameraProcess::CameraProcess(GpuType gpuType) {
		videoProcess = std::make_unique<VideoProcess>(gpuType);
		videoDisplay = std::make_unique<VideoDisplay>();
	}

	CameraProcess::~CameraProcess() {}

	IMattingLayer* CameraProcess::getMattingLayer() {
		if (videoProcess == nullptr || video == nullptr) {
			return nullptr;
		}
		return videoProcess->getMattingLayer();
	}

	bool CameraProcess::setCamera(const char* deviceId, int32_t formatIndex) {
		assert(deviceId);
		this->cameraId = deviceId;
		this->formatIndex = formatIndex;
		video = CameraManager::Get().getVideoDevice(cameraId.c_str());
		if (video == nullptr) {
			return false;
		}
		if (this->formatIndex < 0) {
			this->formatIndex = video->findFormatIndex(1920, 1080);
		}
		videoProcess->initDevice(video, this->formatIndex);
		return true;
	}

	VideoFormat CameraProcess::getVideoFormat() { return video->getSelectFormat(); }

	bool CameraProcess::openCamera(ProcessType processType,
		IOutputLayerObserver* handle) {
		// 如果是开启状态,先关闭
		if (videoProcess->bOpen()) {
			videoProcess->closeDevice();
		}
		// 如果要输出给直播模块
		if ((processType & ProcessType::transport) == ProcessType::transport) {
			videoProcess->getTransportOut()->setObserver(handle);
		}
		// 如果是本地扣像显示模式
		if ((processType & ProcessType::matting) == ProcessType::matting) {
			videoDisplay->setOutLayer(videoProcess->getMattingOut());
		}
		// 如果是原图显示模式
		if ((processType & ProcessType::source) == ProcessType::source) {
			videoDisplay->setOutLayer(videoProcess->getSourceOut());
		}
		return videoProcess->openDevice(processType);
	}

	void CameraProcess::closeCamera() {
		if (videoProcess && videoProcess->bOpen()) {
			videoProcess->closeDevice();
		}
	}

	void CameraProcess::updateMatting(const MattingParamet& mp) {
		auto* layer = getMattingLayer();
		if (layer != nullptr) {
			layer->updateParamet(mp);
		}
	}
	void CameraProcess::updateOperate(const TexOperateParamet& op) {
		if (videoProcess) {
			videoProcess->operateLayer->updateParamet(op);
		}
	}

}  // namespace aoce