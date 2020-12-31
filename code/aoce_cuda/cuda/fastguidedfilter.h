#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_common.h"
#include "helper_math.h"
#include "CudaTypes.hpp"
//#include <device_functions.h>

// 这几个文件只用于nvcc编译,不会污染别的头文件
using namespace aoce::cuda;

inline __global__ void combin(PtrStepSz<uchar3> source, PtrStepSz<uchar> gray, PtrStepSz<float4> dest) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.width && idy < source.height) {
		dest(idy, idx) = rgbauchar42float4(make_uchar4(source(idy, idx).x, source(idy, idx).y, source(idy, idx).z, gray(idy, idx)));
	}
}

inline __global__ void findMatrix(PtrStepSz<float4> source, PtrStepSz<float3> dest, PtrStepSz<float3> dest1, PtrStepSz<float3> dest2) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < source.width && idy < source.height) {
		float4 scolor = source(idy, idx);// rgbauchar42float4(source(idy, idx));
		float3 color = make_float3(scolor);

		dest(idy, idx) = color * scolor.w;
		dest1(idy, idx) = color.x*color;
		dest2(idy, idx) = make_float3(color.y*color.y, color.y*color.z, color.z*color.z);
	}
}

//导向滤波求值 Guided filter 论文地址http://kaiminghe.com/publications/pami12guidedfilter.pdf
//https://blog.csdn.net/baimafujinji/article/details/74750283
inline __global__ void guidedFilter(PtrStepSz<float4> source, PtrStepSz<float3> col1, PtrStepSz<float3> col2, PtrStepSz<float3> col3, PtrStepSz<float4> dest, float eps) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < source.width && idy < source.height) {
		float4 color = source(idy, idx);// rgbauchar42float4(source(idy, idx));
		float3 mean_I = make_float3(color);
		float mean_p = color.w;
		float3 mean_Ip = col1(idy, idx);// rgbauchar32float3(col1(idy, idx));
		float3 var_I_r = col2(idy, idx) - mean_I.x*mean_I;// rgbauchar32float3(col2(idy, idx)) - mean_I.x*mean_I;//col0
		float3 var_I_gbxfv = col3(idy, idx);// rgbauchar32float3(col3(idy, idx));
		//计算方差
		float gg = var_I_gbxfv.x - mean_I.y*mean_I.y;
		float gb = var_I_gbxfv.y - mean_I.y*mean_I.z;
		float bb = var_I_gbxfv.z - mean_I.z*mean_I.z;
		//cov为协方差
		float3 cov_Ip = mean_Ip - mean_I * mean_p;
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
		//把当前ay+by+cz+w的解存入
		dest(idy, idx) = make_float4(a, b);
	}
}

inline __global__ void guidedFilterResult(PtrStepSz<uchar4> source, PtrStepSz<float4> guid, PtrStepSz<uchar4> dest, float intensity) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idx < source.width && idy < source.height) {
		float4 color = rgbauchar42float4(source(idy, idx));//I
		float3 rgb = make_float3(color);
		float4 mean = guid(idy, idx);
		//导向滤波的结果
		float alpha = clamp(color.x*mean.x + color.y*mean.y + color.z*mean.z + mean.w, 0.f, 1.f);
		if (alpha < intensity) {
			rgb = make_float3(0.f, 0.f, 0.f);
			alpha = 0;
		}
		else
			alpha = fminf(alpha + intensity, 1);
		float4 rc = make_float4(rgb, alpha);
		dest(idy, idx) = rgbafloat42uchar4(rc);
	}
}

//分开显示颜色与导向图
inline __global__ void split(PtrStepSz<float4> source, PtrStepSz<uchar4> I, PtrStepSz<uchar> P) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.width && idy < source.height) {
		float4 color = source(idy, idx);
		I(idy, idx) = rgbafloat42uchar4(make_float4(color.z, color.y, color.x, color.w));
		P(idy, idx) = I(idy, idx).w;
	}
}

//深度图对齐成颜色图的大小
inline __global__ void mainAlign(PtrStepSz<uint16_t> source, PtrStepSz<uint16_t> dest, Intrinsics alignParam) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.width  && idy < source.height) {
		float depth = alignParam.alignScale * source(idy, idx);
		if (depth <= 0)
			return;
		//取深度平面的左上角
		float2 depth_pixel = make_float2(idx - 0.5f, idy - 0.5f);
		//在焦点面的UV,intrinF像素平面的焦距，intrinPP像素平面中心点
		float2 xy = (depth_pixel - alignParam.intrinPP) / alignParam.intrinF;
		//得到深度空间下的位置信息
		float3 depth_point = make_float3(xy*depth, depth);
		//转到颜色空间下对应的位置信息
		float3 other_point = mulMat(depth_point, alignParam.m1, alignParam.m2, alignParam.m3) + alignParam.translation;
		//深度空间转到颜色像素平面
		float2 pxy = make_float2(other_point.x, other_point.y) / other_point.z;
		float2 other_pixel = pxy * alignParam.otherIntrinF + alignParam.otherIntrinPP;
		//颜色像素平面的坐标位置
		int2 other_xy0 = make_int2(other_pixel.x + 0.5f, other_pixel.y + 0.5f);

		//操作如同上面,取深度平面的右下角
		depth_pixel = make_float2(idx + 0.5, idy + 0.5f);
		xy = (depth_pixel - alignParam.intrinPP) / alignParam.intrinF;
		depth_point = make_float3(xy*depth, depth);
		other_point = mulMat(depth_point, alignParam.m1, alignParam.m2, alignParam.m3) + alignParam.translation;
		pxy = make_float2(other_point.x, other_point.y) / other_point.z;
		other_pixel = pxy * alignParam.otherIntrinF + alignParam.otherIntrinPP;
		int2 other_xy1 = make_int2(other_pixel.x + 0.5f, other_pixel.y + 0.5f);

		int newDepth = source(idy, idx);
		if (other_xy0.x >= 0 && other_xy0.y >= 0 && other_xy1.x < dest.width && other_xy1.y < dest.height) {
			//深度平面对应扩展后颜色像素平面的uv位置与值
			for (int y = other_xy0.y; y < other_xy1.y; ++y)
				for (int x = other_xy0.x; x < other_xy1.x; ++x) {
					int oldDepth = dest(y, x);
					dest(y, x) = oldDepth ? min(oldDepth, newDepth) : newDepth;
				}
		}
	}
}

//inline __global__ void textureOperate(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, TextureOperate toperate)
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
//		uchar4 mapcolor = make_uchar4(rgba[ir], rgba[ig], rgba[ib], rgba[ia]);
//		int xmin = min(toperate.left, 1.f - toperate.right)*source.width;
//		int xmax = max(toperate.left, 1.f - toperate.right)*source.width;
//		int ymin = min(toperate.top, 1.f - toperate.bottom)*source.height;
//		int ymax = max(toperate.top, 1.f - toperate.bottom)*source.height;
//		if (ix < xmin || ix > xmax || iy < ymin || iy > ymax)
//		{
//			mapcolor = make_uchar4(0, 0, 0, 0);
//		}
//		dest(idy, idx) = mapcolor;
//	}
//}

inline __global__ void distortion(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, PtrStepSz<float2> map) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.width && idy < source.height) {
		float2 mapuv = map(idy, idx);
		float src_x = mapuv.x*source.width;
		float src_y = mapuv.y*source.height;
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