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

		dst(dst_y, dst_x) = src(__float2int_rz(src_y), __float2int_rz(src_x));
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