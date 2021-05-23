#include "CameraManager.hpp"

namespace aoce {

	CameraManager *CameraManager::instance = nullptr;

	CameraManager &CameraManager::Get() {
		if (instance == nullptr) {
			instance = new CameraManager();
		}
		return *instance;
	}

	CameraManager::CameraManager(/* args */) {
#if WIN32
		deviceList.clear();
		auto mfdeviceCount = getVideoManager(CameraType::win_mf)->getDeviceCount();
		if (mfdeviceCount > 0) {
			std::vector<IVideoDevice *> mfdevices(mfdeviceCount);
			getVideoManager(CameraType::win_mf)
				->getDevices(mfdevices.data(), mfdeviceCount);
			for (auto &device : mfdevices) {
				std::string string = device->getName();
				if (string.find("Intel(R) RealSense(TM)") != string.npos) {
					continue;
				}
				deviceList.push_back(device);
			}
		}
		auto rldeviceCount =
			getVideoManager(CameraType::realsense)->getDeviceCount();
		if (rldeviceCount > 0) {
			std::vector<IVideoDevice *> rldevices(rldeviceCount);
			getVideoManager(CameraType::realsense)
				->getDevices(rldevices.data(), rldeviceCount);
			for (auto &device : rldevices) {
				deviceList.push_back(device);
			}
		}
#elif __ANDROID__
		auto andCount = getVideoManager(CameraType::and_camera2)->getDeviceCount();
		std::vector<IVideoDevice *> andevices(andCount);
		if (andCount > 0) {
			getVideoManager(CameraType::and_camera2)
				->getDevices(andevices.data(), andCount);
			// 只添加后置摄像头
			for (auto &device : rldevices) {
				if (device->back()) {
					deviceList.push_back(device);
				}
			}
		}
#endif
	}

	CameraManager::~CameraManager() {}

	// C++11之后返回std::vector会优化,调用拷贝消除
	std::vector<IVideoDevice *> CameraManager::getVideoList() { return deviceList; }

	IVideoDevice *CameraManager::getVideoDevice(const char *deviceId) {
		for (auto device : deviceList) {
			if (strcmp(deviceId, device->getId()) == 0) {
				return device;
			}
		}
		return nullptr;
	}

	}  // namespace aoce
