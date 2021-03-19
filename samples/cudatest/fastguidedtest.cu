#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda/cuda_common.h>
#include <cuda/helper_math.h>
#include "aoce/Aoce.hpp"


using namespace cv;
using namespace cv::cuda;

__global__ void combin(PtrStepSz<uchar3> source, PtrStepSz<uchar> gray, PtrStepSz<float4> dest){
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)	{
		dest(idy, idx) = rgbauchar42float4(make_uchar4(source(idy, idx).x, source(idy, idx).y, source(idy, idx).z, gray(idy, idx)));
	}
}

__global__ void findMatrix(PtrStepSz<float4> source, PtrStepSz<float3> dest, PtrStepSz<float3> dest1, PtrStepSz<float3> dest2){
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < source.cols && idy < source.rows)
	{
		float4 scolor = source(idy, idx);// rgbauchar42float4(source(idy, idx));
		float3 color = make_float3(scolor);

		dest(idy, idx) = color*scolor.w;
		dest1(idy, idx) = color.x*color;
		dest2(idy, idx) = make_float3(color.y*color.y, color.y*color.z, color.z*color.z);
	}
}

//导向滤波求值 Guided filter 论文地址http://kaiminghe.com/publications/pami12guidedfilter.pdf
//https://blog.csdn.net/baimafujinji/article/details/74750283
inline __global__ void guidedFilter(PtrStepSz<float4> source, PtrStepSz<float3> col1, PtrStepSz<float3> col2, PtrStepSz<float3> col3, PtrStepSz<float4> dest, float eps){
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < source.cols && idy < source.rows)	{
		float4 color = source(idy, idx);// rgbauchar42float4(source(idy, idx));
		float3 mean_I = make_float3(color);
		float mean_p = color.w;
		float3 mean_Ip = col1(idy, idx);// rgbauchar32float3(col1(idy, idx));
		float3 var_I_r = col2(idy, idx) - mean_I.x*mean_I;// rgbauchar32float3(col2(idy, idx)) - mean_I.x*mean_I;//col0
		float3 var_I_gbxfv = col3(idy, idx);// rgbauchar32float3(col3(idy, idx));
		float gg = var_I_gbxfv.x - mean_I.y*mean_I.y;
		float gb = var_I_gbxfv.y - mean_I.y*mean_I.z;
		float bb = var_I_gbxfv.z - mean_I.z*mean_I.z;

		float3 cov_Ip = mean_Ip - mean_I*mean_p;
		float3 col0 = var_I_r + make_float3(eps, 0.f, 0.f);
		float3 col1 = make_float3(var_I_r.y, gg + eps, gb);
		float3 col2 = make_float3(var_I_r.z, gb, bb + eps);

		float3 invCol0 = make_float3(1.f, 0.f, 0.f);
		float3 invCol1 = make_float3(0.f, 1.f, 0.f);
		float3 invCol2 = make_float3(0.f, 0.f, 1.f);
		inverseMat3x3(col0, col1, col2, invCol0, invCol1, invCol2);
		//ax+b,a,b是其线性关系解
		float3 a = mulMat(cov_Ip, invCol0, invCol1, invCol2);
		float b = mean_p - dot(a, mean_I);

		dest(idy, idx) = make_float4(a, b);
	}
}

__global__ void guidedFilterResult(PtrStepSz<float4> source, PtrStepSz<float4> guid, PtrStepSz<uchar4> dest, PtrStepSz<uchar> destp){
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < source.cols && idy < source.rows)	{
		float4 color = source(idy, idx);// rgbauchar42float4(source(idy, idx));//I
		float4 mean = guid(idy, idx);
		float alpha = clamp(color.x*mean.x + color.y*mean.y + color.z*mean.z + mean.w, 0.f, 1.f);
		float3 rgb = make_float3(color*alpha);
		dest(idy, idx) = rgbafloat42uchar4(make_float4(rgb, alpha));
		destp(idy, idx) = (uchar)(__saturatef(alpha)*255.0f);
	}
}

extern "C" void testFastGuided(){

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

	std::string imgPathI = getAocePath()+ "/images/toy.bmp";
	std::string imgPathP = getAocePath()+ "/images/toy-mask.bmp";
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

	while (int key = cv::waitKey(1))	{
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
		resultIP.download(cpuIP);
		resultP.download(cpuP);
		cv::imshow(windowNameIP, cpuIP);		
		cv::imshow(windowNameP, cpuP);
	}
}