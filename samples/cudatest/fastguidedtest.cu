#include "cudatest.h"

extern "C" void testFastGuided() {

	cv::cuda::setDevice(0);

	std::string windowNameIP = "vvvvvIP";
	namedWindow(windowNameIP);
	std::string windowNameP = "vvvvvP";
	namedWindow(windowNameP);

	Stream curentStream = {};
	cudaStream_t cudaStream = StreamAccessor::getStream(curentStream);

	int scale = 8;
	int width = 556;
	int height = 568;
	int scaleWidth = width / scale;
	int scaleHeight = height / scale;

	Mat frame(height, width, CV_8UC3);
	Mat resultFrame;// (height, width, CV_8UC3);
	Mat cpuIP;// (scaleHeight, scaleWidth, CV_8UC4);
	Mat cpuP;

	Mat show1;
	Mat show2;

	Mat I(height, width, CV_8UC3);
	Mat P(height, width, CV_8UC3);
	cv::cuda::GpuMat gpuI;
	cv::cuda::GpuMat gpuP;

	cv::cuda::GpuMat gpuKeying(height, width, CV_32FC4);
	cv::cuda::GpuMat gpuCvting;
	//I_sub+p_sub
	cv::cuda::GpuMat gpuResize(scaleHeight, scaleWidth, CV_32FC4);//I_sub+p_sub																 
	cv::cuda::GpuMat mean_I(scaleHeight, scaleWidth, CV_32FC4); //box I_sub+p_sub  mean_Irgb+mean_p

	cv::cuda::GpuMat mean_Ipv(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_rxv(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_gbxfv(scaleHeight, scaleWidth, CV_32FC3);

	cv::cuda::GpuMat mean_Ip(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_rx(scaleHeight, scaleWidth, CV_32FC3);
	cv::cuda::GpuMat var_I_gbxf(scaleHeight, scaleWidth, CV_32FC3);

	cv::cuda::GpuMat meanv(scaleHeight, scaleWidth, CV_32FC4);
	cv::cuda::GpuMat means(scaleHeight, scaleWidth, CV_32FC4);
	cv::cuda::GpuMat mean(scaleHeight, scaleWidth, CV_32FC4);
	cv::cuda::GpuMat resultIP(height, width, CV_8UC4);
	cv::cuda::GpuMat resultP(height, width, CV_8UC1);

	std::string aocePath = aoce::getAocePath();

	std::string imgPathI = aocePath + "/assets/images/top.bmp";//lookup_amatorka  toy.bmp
	std::string imgPathP = aocePath + "/assets/images/toy-mask.bmp";
	I = cv::imread(imgPathI.c_str(), IMREAD_COLOR);	
	P = cv::imread(imgPathP.c_str(), IMREAD_GRAYSCALE);

	dim3 block(32, 4);
	dim3 grid(divUp(width, block.x), divUp(height, block.y));
	dim3 grid2(divUp(scaleWidth, block.x), divUp(scaleHeight, block.y));

	//创建blue	
	auto filter = cv::cuda::createBoxFilter(CV_8UC4, CV_8UC4, Size(3, 3));//包装的NPP里的nppiFilterBox_8u_C4R
	int softness = 21;
	float eps = 0.000001f;
	NppiSize oSizeROI; //NPPI blue
	oSizeROI.width = scaleWidth;
	oSizeROI.height = scaleHeight;
	NppiSize oMaskSize;
	oMaskSize.height = softness;
	oMaskSize.width = softness;
	NppiPoint oAnchor;
	oAnchor.x = oMaskSize.width / 2;
	oAnchor.y = oMaskSize.height / 2;
	NppiPoint oSrcOffset = { 0, 0 };

	while (int key = cv::waitKey(1)) {
		gpuI.upload(I);
		gpuP.upload(P);
		//把颜色通道与导向通道合并,他们很多计算是同样的,合成四通道的加速并容易对齐32/64/128这些值
		combin << <grid, block, 0, cudaStream >> > (gpuI, gpuP, gpuKeying);
		//导向滤波这算法的优势,与图像大小可以做到无关,在这我们使用缩小8倍后的大小
		cv::cuda::resize(gpuKeying, gpuResize, cv::Size(scaleWidth, scaleHeight), 0, 0, cv::INTER_NEAREST, curentStream);
		//计算矩阵  rr, rg, rb/rg, gg, gb/rb, gb, bb
		findMatrix << <grid2, block, 0, cudaStream >> > (gpuResize, mean_Ipv, var_I_rxv, var_I_gbxfv);
		//模糊缩小后的原始值
		nppiFilterBoxBorder_32f_C4R((Npp32f*)gpuResize.ptr<float4>(), gpuResize.step, oSizeROI, oSrcOffset, (Npp32f*)mean_I.ptr<float4>(), mean_I.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		//模糊矩阵
		nppiFilterBoxBorder_32f_C3R((Npp32f*)mean_Ipv.ptr<float3>(), mean_Ipv.step, oSizeROI, oSrcOffset, (Npp32f*)mean_Ip.ptr<float3>(), mean_Ip.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		nppiFilterBoxBorder_32f_C3R((Npp32f*)var_I_rxv.ptr<float3>(), var_I_rxv.step, oSizeROI, oSrcOffset, (Npp32f*)var_I_rx.ptr<float3>(), var_I_rx.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		nppiFilterBoxBorder_32f_C3R((Npp32f*)var_I_gbxfv.ptr<float3>(), var_I_gbxfv.step, oSizeROI, oSrcOffset, (Npp32f*)var_I_gbxf.ptr<float3>(), var_I_gbxf.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		//求导
		guidedFilter << <grid2, block, 0, cudaStream >> > (mean_I, mean_Ip, var_I_rx, var_I_gbxf, meanv, eps);
		//模糊求导的结果
		nppiFilterBoxBorder_32f_C4R((Npp32f*)meanv.ptr<float4>(), meanv.step, oSizeROI, oSrcOffset, (Npp32f*)means.ptr<float4>(), means.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		//返回到原图像大小
		cv::cuda::resize(means, mean, cv::Size(width, height), 0, 0, cv::INTER_LINEAR, curentStream);
		//求结果
		guidedFilterResult << <grid, block, 0, cudaStream >> > (gpuKeying, mean, resultIP, resultP);
		//显示结果
		meanv.download(show1);
		means.download(show2);
		resultIP.download(cpuIP);
		resultP.download(cpuP);
		cv::imshow(windowNameIP, cpuIP);
		cv::imshow(windowNameP, cpuP);
	}
}