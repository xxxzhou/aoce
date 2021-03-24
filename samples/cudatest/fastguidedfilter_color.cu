#include "cudatest.h"

static Keying keyParamet = {};
static int softness = 10;
static float eps = 0.000001f;

extern "C" void test10() {
    keyParamet.chromaColor = {0.15f, 0.6f, 0.0f};
    keyParamet.alphaScale = 20.0f;
    keyParamet.alphaExponent = 0.1f;
    keyParamet.alphaCutoffMin = 0.2f;
    keyParamet.lumaMask = 2.0f;
    keyParamet.ambientColor = {0.1f, 0.1f, 0.9f};
    keyParamet.despillScale = 0.0f;
    keyParamet.despillExponent = 0.1f;
    keyParamet.ambientScale = 1.0f;

	std::string windowNameResult = "result";
	std::string windowNameI = "I";
	std::string windowNameP = "P";
	std::string uiName = "uivvvvvvvvvvvv";
	namedWindow(windowNameResult);
	namedWindow(windowNameI);
	namedWindow(windowNameP);
	// crateUI1(uiName.c_str());
	//StreamAccessor::getStream/wrapStream
	Stream curentStream = {};
	cudaStream_t cudaStream = StreamAccessor::getStream(curentStream);
	//nppSetStream(cudaStream);
	VideoCapture cap = {};
	int scale = 16;
	int width = 1280;
	int height = 720;
	int scaleWidth = width / scale;
	int scaleHeight = height / scale;

	Mat frame(height, width, CV_8UC3);
	Mat cpuResult;
	Mat cpuGpuKeying;

	Mat cpuI(height, width, CV_8UC4);
	Mat cpuP(height, width, CV_8UC1);
	cv::cuda::GpuMat gpuI(height, width, CV_8UC4);;
	cv::cuda::GpuMat gpuP(height, width, CV_8UC1);;

	cv::cuda::GpuMat gpuFrame;// (height, width, CV_8UC3);
	cv::cuda::GpuMat sourceFrame(height, width, CV_8UC4);
	cv::cuda::GpuMat gpuKeying(height, width, CV_32FC4);
	cv::cuda::GpuMat gpuCvting;
	//I_sub+p_sub
	cv::cuda::GpuMat gpuResize(scaleHeight, scaleWidth, CV_32FC4);//I_sub+p_sub
	//box I_sub+p_sub  mean_Irgb+mean_p
	cv::cuda::GpuMat mean_I(scaleHeight, scaleWidth, CV_32FC4);

	cv::cuda::GpuMat mean_Ipv(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_rxv(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_gbxfv(scaleHeight, scaleWidth, CV_32FC3);

	cv::cuda::GpuMat mean_Ip(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_rx(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_gbxf(scaleHeight, scaleWidth, CV_32FC3);

	cv::cuda::GpuMat meanv(scaleHeight, scaleWidth, CV_32FC4);
	cv::cuda::GpuMat means(scaleHeight, scaleWidth, CV_32FC4);
	cv::cuda::GpuMat mean(scaleHeight, scaleWidth, CV_32FC4);
	//cv::cuda::GpuMat result(height, width, CV_8UC4);
	cv::cuda::GpuMat resultIP(height, width, CV_8UC4);
	cv::cuda::GpuMat resultP(height, width, CV_8UC1);

	dim3 block(32, 4);
	dim3 grid(divUp(width, block.x), divUp(height, block.y));
	dim3 grid2(divUp(scaleWidth, block.x), divUp(scaleHeight, block.y));

	NppiSize oSizeROI; //NPPI blue
	oSizeROI.width = scaleWidth;
	oSizeROI.height = scaleHeight;
	NppiSize oMaskSize = {};
	oMaskSize.height = softness;
	oMaskSize.width = softness;
	NppiPoint oAnchor = {};
	oAnchor.x = oMaskSize.width / 2;
	oAnchor.y = oMaskSize.height / 2;
	NppiPoint oSrcOffset = { 0, 0 };

	cap.open(0, CAP_DSHOW);
	cap.set(CAP_PROP_FRAME_WIDTH, width);
	cap.set(CAP_PROP_FRAME_HEIGHT, height);
	cap.set(CAP_PROP_FPS, 30);
	auto ctime = clock();
	auto pretime = clock();
	int softness = 1;

	while (int key = cv::waitKey(1)) {
		if (curentStream.queryIfComplete()) {
			// updateUI1(uiName.c_str());
		}
		cap >> frame;
		ctime = clock();
		gpuFrame.upload(frame, curentStream);
		cv::cuda::cvtColor(gpuFrame, sourceFrame, COLOR_BGR2RGBA, 0, curentStream);
		simpleKeyingUE4 << <grid, block, 0, cudaStream >> > (sourceFrame, gpuKeying, keyParamet);		
		gpuKeying.download(cpuGpuKeying);
		cv::cvtColor(cpuGpuKeying, cpuGpuKeying, COLOR_BGR2RGBA);
		//缩放大小
		cv::cuda::resize(gpuKeying, gpuResize, cv::Size(scaleWidth, scaleHeight), 0, 0, cv::INTER_NEAREST, curentStream);

		findMatrix << <grid2, block, 0, cudaStream >> > (gpuResize, mean_Ipv, var_I_rxv, var_I_gbxfv);
		//此外，在计算每个窗口的线性系数时，我们可以发现一个像素会被多个窗口包含，也就是说，每个像素都由多个线性函数所描述。
		//因此，如之前所说，要具体求某一点的输出值时，只需将所有包含该点的线性函数值平均即可
		nppiFilterBoxBorder_32f_C4R((Npp32f*)gpuResize.ptr<float4>(), gpuResize.step, oSizeROI, oSrcOffset, (Npp32f*)mean_I.ptr<float4>(), mean_I.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		nppiFilterBoxBorder_32f_C3R((Npp32f*)mean_Ipv.ptr<float3>(), mean_Ipv.step, oSizeROI, oSrcOffset, (Npp32f*)mean_Ip.ptr<float3>(), mean_Ip.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		nppiFilterBoxBorder_32f_C3R((Npp32f*)var_I_rxv.ptr<float3>(), var_I_rxv.step, oSizeROI, oSrcOffset, (Npp32f*)var_I_rx.ptr<float3>(), var_I_rx.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		nppiFilterBoxBorder_32f_C3R((Npp32f*)var_I_gbxfv.ptr<float3>(), var_I_gbxfv.step, oSizeROI, oSrcOffset, (Npp32f*)var_I_gbxf.ptr<float3>(), var_I_gbxf.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		//求导
		guidedFilter << <grid2, block, 0, cudaStream >> > (mean_I, mean_Ip, var_I_rx, var_I_gbxf, meanv, eps);
		nppiFilterBoxBorder_32f_C4R((Npp32f*)meanv.ptr<float4>(), meanv.step, oSizeROI, oSrcOffset, (Npp32f*)means.ptr<float4>(), means.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		meanv.download(cpuI);
		means.download(cpuP);
		cv::cvtColor(cpuI, cpuI, COLOR_BGR2RGBA);
		cv::cvtColor(cpuP, cpuP, COLOR_BGR2RGBA);
		cv::cuda::resize(means, mean, cv::Size(width, height), 0, 0, cv::INTER_LINEAR, curentStream);
		//求结果
		guidedFilterResult << <grid, block, 0, cudaStream >> > (gpuKeying, mean, resultIP, resultP);
		//gpuKeying分别显示三通与一通
		//split << <grid, block, 0, cudaStream >> > (gpuKeying, gpuI, gpuP);
		cv::cuda::cvtColor(resultIP, gpuCvting, COLOR_BGRA2RGBA, 0, curentStream);
		gpuCvting.download(cpuResult);

		cv::imshow(windowNameResult, cpuResult);
		cv::imshow(windowNameI, cpuI);
		cv::imshow(windowNameP, cpuP);
	}
	//cudaFree((void*)guideData);
	//cudaFree((void*)guideDataCopy);
}