#pragma once
#include <Aoce.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaTypes.hpp"
#include "cuda_common.h"

// 这几个文件只用于nvcc编译,不会污染别的头文件
using namespace aoce;
using namespace aoce::cuda;

//copy opencv resize实现，因为引入opencv_cudawarping太大
template <typename T> __global__
void resize_nearest(const PtrStep<T> src, PtrStepSz<T> dst, const float fx, const float fy) {
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < dst.width && dst_y < dst.height) {
		const float src_x = dst_x * fx;
		const float src_y = dst_y * fy;

		T out = src(__float2int_rz(src_y), __float2int_rz(src_x));

		dst(dst_y, dst_x) = out;
	}
}

template <typename T> __global__
void resize_linear(const PtrStepSz<T> src, PtrStepSz<T> dst, const float fx, const float fy) {
	// typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < dst.width && dst_y < dst.height) {
		const float src_x = dst_x * fx;
		const float src_y = dst_y * fy;

		// work_type out = VecTraits<work_type>::all(0);
        T out = {};

		const int x1 = __float2int_rd(src_x);
		const int y1 = __float2int_rd(src_y);
		const int x2 = x1 + 1;
		const int y2 = y1 + 1;
		const int x2_read = ::min(x2, src.width - 1);
		const int y2_read = ::min(y2, src.height - 1);

		T src_reg = src(y1, x1);
		out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

		src_reg = src(y1, x2_read);
		out = out + src_reg * ((src_x - x1) * (y2 - src_y));

		src_reg = src(y2_read, x1);
		out = out + src_reg * ((x2 - src_x) * (src_y - y1));

		src_reg = src(y2_read, x2_read);
		out = out + src_reg * ((src_x - x1) * (src_y - y1));

		dst(dst_y, dst_x) = out;
	}
}

// inline __global__ void textureOperate(PtrStepSz<uchar4> source,
// PtrStepSz<uchar4> dest, TextureOperate toperate)
//{
//	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
//	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (idx < source.width && idy < source.height)
//	{
//		int ix = idx;
//		int iy = idy;
//		if (toperate.bFlipX)
//		{
//			ix = source.width - idx;
//		}
//		if (toperate.bFlipY)
//		{
//			iy = source.height - idy;
//		}
//		uchar4 color = source(iy, ix);
//		uchar rgba[4] = { color.x,color.y,color.z,color.w };
//		int ir = clamp(toperate.mapR, 0, 3);
//		int ig = clamp(toperate.mapG, 0, 3);
//		int ib = clamp(toperate.mapB, 0, 3);
//		int ia = clamp(toperate.mapA, 0, 3);
//		uchar4 mapcolor = make_uchar4(rgba[ir], rgba[ig], rgba[ib],
//rgba[ia]); 		int xmin = min(toperate.left, 1.f - toperate.right)*source.width;
//		int xmax = max(toperate.left, 1.f -
//toperate.right)*source.width; 		int ymin = min(toperate.top, 1.f -
//toperate.bottom)*source.height; 		int ymax = max(toperate.top, 1.f -
//toperate.bottom)*source.height; 		if (ix < xmin || ix > xmax || iy < ymin || iy
//> ymax)
//		{
//			mapcolor = make_uchar4(0, 0, 0, 0);
//		}
//		dest(idy, idx) = mapcolor;
//	}
//}

inline __global__ void distortion(PtrStepSz<uchar4> source,
                                  PtrStepSz<uchar4> dest,
                                  PtrStepSz<float2> map) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < source.width && idy < source.height) {
        float2 mapuv = map(idy, idx);
        float src_x = mapuv.x * source.width;
        float src_y = mapuv.y * source.height;
        const int x1 = clamp(__float2int_rd(src_x), 0, source.width - 1);
        const int y1 = clamp(__float2int_rd(src_y), 0, source.height - 1);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = clamp(x2, 0, source.width - 1);
        const int y2_read = clamp(y2, 0, source.height - 1);

        float4 out = make_float4(0.f);
        float4 src_reg = rgbauchar42float4(source(y1, x1));
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = rgbauchar42float4(source(y1, x2_read));
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = rgbauchar42float4(source(y2_read, x1));
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = rgbauchar42float4(source(y2_read, x2_read));
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        dest(idy, idx) = rgbafloat42uchar4(out);
    }
}