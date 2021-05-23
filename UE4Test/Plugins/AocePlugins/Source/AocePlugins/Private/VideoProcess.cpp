#include "VideoProcess.hpp"

// using namespace std::placeholders;

namespace aoce {

	VideoProcess::VideoProcess(GpuType gpuType) {
		this->gpuType = gpuType;
#if WIN32
		if (gpuType == GpuType::other) {
			this->gpuType = GpuType::cuda;
		}
#elif __ANDROID__
		this->gpuType = GpuType::vulkan;
#endif
		// 生成一张执行图
		graph = getPipeGraphFactory(this->gpuType)->createGraph();
		auto* layerFactory = getLayerFactory(this->gpuType);
		inputLayer = std::unique_ptr<IInputLayer>(layerFactory->crateInput());
		inputLayer1 = std::unique_ptr<IInputLayer>(layerFactory->crateInput());
		outputLayer = std::unique_ptr<IOutputLayer>(layerFactory->createOutput());
		outputLayer1 = std::unique_ptr<IOutputLayer>(layerFactory->createOutput());
		outputLayer2 = std::unique_ptr<IOutputLayer>(layerFactory->createOutput());
		yuv2rgbLayer =
			std::unique_ptr<IYUV2RGBALayer>(layerFactory->createYUV2RGBA());
		rgb2yuvLayer =
			std::unique_ptr<IRGBA2YUVLayer>(layerFactory->createRGBA2YUV());

		outputLayer->updateParamet({ true, false });
		outputLayer1->updateParamet({ false, true });
		outputLayer2->updateParamet({ false, true });
		rgb2yuvLayer->updateParamet({ VideoType::yuy2P, true });
	}

	VideoProcess::~VideoProcess() {}

	void VideoProcess::initDevice(IVideoDevice* videoPtr, int32_t formatIndex) {
		this->video = videoPtr;
		if (formatIndex < 0) {
			formatIndex = video->findFormatIndex(1920, 1080);
		}
		video->setFormat(formatIndex);
		format = this->video->getSelectFormat();
		video->setObserver(this);
		mattingLayer = std::unique_ptr<IMattingLayer>(
			createMattingLayer(gpuType, video->bDepth()));
		// 清空graph里的节点与线
		graph->clear();
		// outputLayer用于传输出去
		graph->addNode(inputLayer.get())
			->addNode(yuv2rgbLayer.get())
			->addNode(mattingLayer.get())
			->addNode(rgb2yuvLayer.get())
			->addNode(outputLayer.get());
		graph->addNode(outputLayer1.get());
		graph->addNode(outputLayer2.get());
		if (video->bDepth()) {
			graph->addNode(inputLayer1.get());
			inputLayer1->getLayer()->addLine(mattingLayer->getLayer(), 0, 1);
		}
		// outputLayer1输出原图
		yuv2rgbLayer->getLayer()->addLine(outputLayer1->getLayer());
		// outputLayer2输出扣像图
		mattingLayer->getLayer()->addLine(outputLayer2->getLayer());
	}

	bool VideoProcess::bOpen() {
		if (video) {
			MattingParamet mp = {};
			mp.colorParamet.average = 0.95;
			mp.colorParamet.percent = 0.95;
			mp.colorParamet.bpercent = 0.84;
			mp.colorParamet.bBlueKey = 0;
			mp.colorParamet.saturation = 0.97;
			mp.colorParamet.greenRemove = 0.0078;
			mp.colorParamet.intensity = 2;
			mp.colorParamet.gamma = 1.0f;
			mp.colorParamet.amount = 1.0f;
			mp.colorParamet.softness = 12;
			mp.colorParamet.spread = 2;
			mp.colorParamet.eps = 0.00001f;
			mattingLayer->updateParamet(mp);
			return video->bOpen();
		}
		return false;
	}

	VideoFormat VideoProcess::getVideoFormat() { return this->format; }

	bool VideoProcess::openDevice(ProcessType processType) {
		this->processType = processType;
		bool bSourceOutput =
			((processType & ProcessType::source) == ProcessType::source);
		bool bMattingOutput =
			((processType & ProcessType::matting) == ProcessType::matting);
		bool bTransport =
			((processType & ProcessType::transport) == ProcessType::transport);
		outputLayer->getLayer()->setVisable(bTransport);
		outputLayer1->getLayer()->setVisable(bSourceOutput);
		outputLayer2->getLayer()->setVisable(bMattingOutput);
		return video->open();
	}

	void VideoProcess::closeDevice() {
		outputLayer->setObserver(nullptr);
		outputLayer1->setObserver(nullptr);
		outputLayer2->setObserver(nullptr);
		video->close();
	}

	void VideoProcess::onVideoFrame(VideoFrame frame) {
		if (getYuvIndex(frame.videoType) >= 0) {
			yuv2rgbLayer->getLayer()->setVisable(true);
			if (yuv2rgbLayer->getParamet().type != frame.videoType) {
				yuv2rgbLayer->updateParamet({ frame.videoType, true });
			}
		}
		else {
			inputLayer->getLayer()->addLine(outputLayer1->getLayer());
			yuv2rgbLayer->getLayer()->setVisable(false);
		}
		inputLayer->inputCpuData(frame);
		graph->run();
	}

	void VideoProcess::onDepthVideoFrame(VideoFrame frame, VideoFrame depth,
		void* alignParamet) {
		inputLayer1->inputCpuData(depth);
		mattingLayer->updateAlignParamet((float*)alignParamet, frame.width,
			frame.height);
		onVideoFrame(frame);
	}

}  // namespace aoce