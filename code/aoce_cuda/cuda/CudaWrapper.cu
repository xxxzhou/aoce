#include "colorconvert.h"
#include "imageprocess.h"

// nvcc与C++编译的转接文件
#define BLOCK_X 32
#define BLOCK_Y 8

// 这几个文件只用于nvcc编译,不会污染别的头文件
// using namespace aoce;
// using namespace aoce::cuda;
namespace aoce {
namespace cuda {

const dim3 block = dim3(BLOCK_X, BLOCK_Y);

void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest, cudaStream_t stream) {
	dim3 grid(divUp(dest.width, block.x), divUp(dest.height, block.y));
	rgb2rgba << <grid, block, 0, stream >> > (source, dest);
}

void rgba2bgr_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar3> dest, cudaStream_t stream) {
	dim3 grid(divUp(dest.width, block.x), divUp(dest.height, block.y));
	rgba2bgr << <grid, block, 0, stream >> > (source, dest);
}

void argb2rgba_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, cudaStream_t stream) {
	dim3 grid(divUp(dest.width, block.x), divUp(dest.height, block.y));
	argb2rgba << <grid, block, 0, stream >> > (source, dest);
}

//yuv planer转换成rgb
void yuv2rgb_gpu(PtrStepSz<uchar> source, PtrStepSz<uchar4> dest, int32_t yuvtype, cudaStream_t stream) {
	dim3 grid(divUp(dest.width/2, block.x), divUp(dest.height/2, block.y));
	if (yuvtype == 1){
		yuv2rgb<1> << <grid, block, 0, stream >> > (source, dest);
	}
	else if (yuvtype == 2){
		yuv2rgb<2> << <grid, block, 0, stream >> > (source, dest);	
	}
	else if (yuvtype == 3){
		dim3 grid(divUp(dest.width/2, block.x), divUp(dest.height, block.y));
		yuv2rgb<3> << <grid, block, 0, stream >> > (source, dest);
	}
}

//packed ufront/yfront (yuyv true/true)/(yvyu false/true)/(uyvy true/false)
void yuv2rgb_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool ufront, bool yfront, cudaStream_t stream) {
	dim3 grid(divUp(source.width, block.x), divUp(source.height, block.y));
	int bitx = ufront ? 0 : 2;
	int yoffset = yfront ? 0 : 1;
	yuv2rgb << <grid, block, 0, stream >> > (source, dest, bitx, yoffset);
}

void yuva2rgb_gpu(PtrStepSz<uchar> source, PtrStepSz<uchar4> dest, cudaStream_t stream) {
	dim3 grid(divUp(dest.width/2, block.x), divUp(dest.height/2, block.y));
	yuva2rgb << <grid, block, 0, stream >> > (source, dest);
}

void rgb2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest, int32_t yuvtype, cudaStream_t stream) {	
	dim3 grid(divUp(source.width/2, block.x), divUp(source.height/2, block.y));
	if (yuvtype == 1){
		rgb2yuv<1> << <grid, block, 0, stream >> > (source, dest);
	}
	else if (yuvtype == 2){
		rgb2yuv<2> << <grid, block, 0, stream >> > (source, dest);
	}
	else if (yuvtype == 3){
		dim3 grid(divUp(source.width/2, block.x), divUp(source.height, block.y));
		rgb2yuv<3> << <grid, block, 0, stream >> > (source, dest);
	}
}

//packed ufront/yfront (yuyv true/true)/(yvyu false/true)/(uyvy true/false)
void rgb2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool ufront, bool yfront, cudaStream_t stream) {
	dim3 grid(divUp(dest.width, block.x), divUp(dest.height, block.y));
	int bitx = ufront ? 0 : 2;
	int yoffset = yfront ? 0 : 1;
	rgb2yuv << <grid, block, 0, stream >> > (source, dest, bitx, yoffset);
}

void rgba2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest, cudaStream_t stream){
	dim3 grid(divUp(source.width/2, block.x), divUp(source.height/2, block.y));
	rgba2yuv << <grid, block, 0, stream >> > (source, dest);
}

void textureMap_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, MapChannel paramt, cudaStream_t stream) {
	dim3 grid(divUp(source.width, block.x), divUp(source.height, block.y));
	textureMap << <grid, block, 0, stream >> > (source, dest, paramt);
}

void blend_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> blendTex, PtrStepSz<uchar4> dest,
	int32_t left, int32_t top, float opacity, cudaStream_t stream) {
	dim3 grid(divUp(source.width, block.x), divUp(source.height, block.y));
	blend << <grid, block, 0, stream >> > (source, blendTex, dest, left, top, opacity);
}

void operate_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, Operate paramt, cudaStream_t stream) {
	dim3 grid(divUp(source.width, block.x), divUp(source.height, block.y));
	operate << <grid, block, 0, stream >> > (source, dest, paramt);
}

void uchar2float_gpu(PtrStepSz<uchar4> source, PtrStepSz<float4> dest, cudaStream_t stream){
	dim3 grid(divUp(source.width, block.x), divUp(source.height, block.y));
	uchar2float << <grid, block, 0, stream >> > (source, dest);
}

template <typename T>
void resize_gpu(PtrStepSz<T> source, PtrStepSz<T> dest, bool bLinear, cudaStream_t stream) {
	float fx = static_cast<float>(source.width) / dest.width;
	float fy = static_cast<float>(source.height) / dest.height;
	dim3 grid(divUp(dest.width, block.x), divUp(dest.height, block.y));
	if (bLinear) {
		resize_linear<T> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
	else {
		resize_nearest<T> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
}

//实例化几个
template void resize_gpu<uchar4>(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool bLinear, cudaStream_t stream);
template void resize_gpu<uchar>(PtrStepSz<uchar> source, PtrStepSz<uchar> dest, bool bLinear, cudaStream_t stream);
template void resize_gpu<float4>(PtrStepSz<float4> source, PtrStepSz<float4> dest, bool bLinear, cudaStream_t stream);

	}
}

