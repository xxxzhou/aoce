#pragma once
// 从cuda samples里直接拿的
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include <nppcore.h>
#include <stdint.h>

#include "helper_math.h"

#if (_MSC_VER >= 1915)
#define no_init_all deprecated
#endif

struct Intrinsics {
	float2 intrinPP;
	float2 intrinF;
	float2 otherIntrinPP;
	float2 otherIntrinF;
	float3 m1;
	float3 m2;
	float3 m3;
	float3 translation;
	float alignScale;
};

typedef unsigned char uchar;

#define UINT82FLOAT 0.00392156862745f

inline __device__ float __saturatef(float x) { return clamp(x, 0.f, 1.f); }

inline __device__ float4 rgbaint2float4(unsigned int c) {
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;          //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;   //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;  //  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;  //  /255.0f;
	return rgba;
}

inline __device__ unsigned int rgbafloat42int(float4 rgba) {
	rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return ((unsigned int)(rgba.w * 255.0f) << 24) |
		((unsigned int)(rgba.z * 255.0f) << 16) |
		((unsigned int)(rgba.y * 255.0f) << 8) |
		((unsigned int)(rgba.x * 255.0f));
}

inline __device__ float4 rgbauchar42float4(uchar4 c) {
	float4 rgba;
	rgba.x = c.x * 0.003921568627f;  //  /255.0f;
	rgba.y = c.y * 0.003921568627f;  //  /255.0f;
	rgba.z = c.z * 0.003921568627f;  //  /255.0f;
	rgba.w = c.w * 0.003921568627f;  //  /255.0f;
	return rgba;
}

inline __device__ unsigned char rgbafloat2ucha1(float x) {
	return (unsigned char)(__saturatef(x) * 255.0f);
}

inline __device__ uchar4 rgbafloat42uchar4(float4 c) {
	uchar4 rgba;
	rgba.x = rgbafloat2ucha1(c.x);
	rgba.y = rgbafloat2ucha1(c.y);
	rgba.z = rgbafloat2ucha1(c.z);
	rgba.w = rgbafloat2ucha1(c.w);
	return rgba;
}

inline __device__ float3 rgbauchar32float3(uchar3 c) {
	float3 rgba;
	rgba.x = c.x * 0.003921568627f;  //  /255.0f;
	rgba.y = c.y * 0.003921568627f;  //  /255.0f;
	rgba.z = c.z * 0.003921568627f;  //  /255.0f;
	return rgba;
}

inline __device__ uchar3 rgbafloat32uchar3(float3 c) {
	uchar3 rgba;
	rgba.x = rgbafloat2ucha1(c.x);
	rgba.y = rgbafloat2ucha1(c.y);
	rgba.z = rgbafloat2ucha1(c.z);
	return rgba;
}

inline __host__ __device__ int4 operator*(int4 a, float b) {
	return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __host__ __device__ void inverseMat3x3(const float3& col0,
	const float3& col1,
	const float3& col2,
	float3& invCol0, float3& invCol1,
	float3& invCol2) {
	float det = col0.x * (col1.y * col2.z - col2.y * col1.z) -
		col0.y * (col1.x * col2.z - col1.z * col2.x) +
		col0.z * (col1.x * col2.y - col1.y * col2.x);
	if (det != 0) {
		float invdet = 1.0f / det;
		invCol0.x = (col1.y * col2.z - col2.y * col1.z) * invdet;
		invCol0.y = (col0.z * col2.y - col0.y * col2.z) * invdet;
		invCol0.z = (col0.y * col1.z - col0.z * col1.y) * invdet;
		invCol1.x = (col1.z * col2.x - col1.x * col2.z) * invdet;
		invCol1.y = (col0.x * col2.z - col0.z * col2.x) * invdet;
		invCol1.z = (col1.x * col0.z - col0.x * col1.z) * invdet;
		invCol2.x = (col1.x * col2.y - col2.x * col1.y) * invdet;
		invCol2.y = (col2.x * col0.y - col0.x * col2.y) * invdet;
		invCol2.z = (col0.x * col1.y - col1.x * col0.y) * invdet;
	}
}

inline __host__ __device__ float3 mulMat(const float3 data, const float3& col0,
	const float3& col1,
	const float3& col2) {
	float3 dest;
	dest.x = dot(data, make_float3(col0.x, col1.x, col2.x));
	dest.y = dot(data, make_float3(col0.y, col1.y, col2.y));
	dest.z = dot(data, make_float3(col0.z, col1.z, col2.z));
	return dest;
}

inline __host__ __device__ float4 make_float4(uchar4 color) {
	float4 result;
	result.x = color.x;
	result.y = color.y;
	result.z = color.z;
	result.w = color.w;
	return result;
}

inline __host__ __device__ float3 make_float3(uchar4 color) {
	float3 result;
	result.x = color.x;
	result.y = color.y;
	result.z = color.z;
	return result;
}

inline __device__ float3 sqrt_float3(const float3 data) {
	float3 result;
	result.x = sqrtf(data.x);
	result.y = sqrtf(data.y);
	result.z = sqrtf(data.z);
	return result;
}

inline __device__ float3 yuv2Rgb(float3 yuv) {
	float3 rgb;
	rgb.x = clamp(yuv.x + 1.402f * (yuv.z - 0.5f), 0.f, 1.f);
	rgb.y = clamp(yuv.x - 0.71414f * (yuv.z - 0.5f) - 0.34414f * (yuv.y - 0.5f),
		0.f, 1.f);
	rgb.z = clamp(yuv.x + 1.772f * (yuv.y - 0.5f), 0.f, 1.f);
	return rgb;
}

inline __device__ float3 rgb2Yuv(float3 rgb) {
	float3 yuv;
	// xyz -> yuv
	yuv.x = clamp(0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z, 0.f, 1.f);
	// uv (-0.5,0.5)
	yuv.y =
		clamp(-0.1687 * rgb.x - 0.3313 * rgb.y + 0.5 * rgb.z + 0.5f, 0.f, 1.f);
	yuv.z =
		clamp(0.5 * rgb.x - 0.4187 * rgb.y - 0.0813 * rgb.z + 0.5f, 0.f, 1.f);
	return yuv;
}

inline __device__ int u22u1(int2 uv, int step) { return uv.y * step + uv.x; }

inline __device__ int2 u12u2(int index, int step) {
	int y = index / step;
	int x = index - y * step;
	return make_int2(x, y);
}