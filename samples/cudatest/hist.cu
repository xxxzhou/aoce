
// https://github.com/opencv/opencv/pull/18136/commits/f617f18e46fa556daea060d3c69307567bbc65f7
// buildLutKernel<<<1, 256, 0, stream>>>
__global__ void buildLutKernel(int* hist, unsigned char* lut, int size) {
    __shared__ int warp_smem[8];
    __shared__ int hist_smem[8][33];

#define HIST_SMEM_NO_BANK_CONFLICT(idx) hist_smem[(idx) >> 5][(idx)&31]

    const int tId = threadIdx.x;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Step1 - Find minimum non-zero value in hist and make it zero
    HIST_SMEM_NO_BANK_CONFLICT(tId) = hist[tId];
    int nonZeroIdx = HIST_SMEM_NO_BANK_CONFLICT(tId) > 0 ? tId : 256;

    __syncthreads();

    for (int delta = 16; delta > 0; delta /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        int shflVal = __shfl_down_sync(0xFFFFFFFF, nonZeroIdx, delta);
#else
        int shflVal = __shfl_down(nonZeroIdx, delta);
#endif
        if (laneId < delta) nonZeroIdx = min(nonZeroIdx, shflVal);
    }

    if (laneId == 0) warp_smem[warpId] = nonZeroIdx;

    __syncthreads();

    if (tId < 8) {
        int warpVal = warp_smem[tId];
        for (int delta = 4; delta > 0; delta /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
            int shflVal = __shfl_down_sync(0x000000FF, warpVal, delta);
#else
            int shflVal = __shfl_down(warpVal, delta);
#endif
            if (tId < delta) warpVal = min(warpVal, shflVal);
        }
        if (tId == 0) {
            warp_smem[0] = warpVal;  // warpVal - minimum index
        }
    }

    __syncthreads();

    const int minNonZeroIdx = warp_smem[0];
    const int minNonZeroVal = HIST_SMEM_NO_BANK_CONFLICT(minNonZeroIdx);
    if (minNonZeroVal == size) {
        // This is a special case: the whole image has the same color

        lut[tId] = 0;
        if (tId == minNonZeroIdx) lut[tId] = minNonZeroIdx;
        return;
    }

    if (tId == 0) HIST_SMEM_NO_BANK_CONFLICT(minNonZeroIdx) = 0;

    __syncthreads();

    // Step2 - Inclusive sum
    // Algorithm from GPU Gems 3 (A Work-Efficient Parallel Scan)
    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    // Step2 Phase1 - The Up-Sweep Phase
    for (int delta = 1; delta < 256; delta *= 2) {
        if (tId < 128 / delta) {
            int idx = 255 - 2 * tId * delta;
            HIST_SMEM_NO_BANK_CONFLICT(idx) +=
                HIST_SMEM_NO_BANK_CONFLICT(idx - delta);
        }
        __syncthreads();
    }

    // Step2 Phase2 - The Down-Sweep Phase
    if (tId == 0) HIST_SMEM_NO_BANK_CONFLICT(255) = 0;

    for (int delta = 128; delta >= 1; delta /= 2) {
        if (tId < 128 / delta) {
            int rootIdx = 255 - tId * delta * 2;
            int leftIdx = rootIdx - delta;
            int tmp = HIST_SMEM_NO_BANK_CONFLICT(leftIdx);
            HIST_SMEM_NO_BANK_CONFLICT(leftIdx) =
                HIST_SMEM_NO_BANK_CONFLICT(rootIdx);
            HIST_SMEM_NO_BANK_CONFLICT(rootIdx) += tmp;
        }
        __syncthreads();
    }

    // Step2 Phase3 - Convert exclusive sum to inclusive sum
    int tmp = HIST_SMEM_NO_BANK_CONFLICT(tId);
    __syncthreads();
    if (tId >= 1) HIST_SMEM_NO_BANK_CONFLICT(tId - 1) = tmp;
    if (tId == 255) HIST_SMEM_NO_BANK_CONFLICT(tId) = tmp + hist[tId];
    __syncthreads();

    // Step3 - Scale values to build lut

    // lut[tId] = saturate_cast<unsigned char>(HIST_SMEM_NO_BANK_CONFLICT(tId) *
    //                                         (255.0f / (size - minNonZeroVal)));

#undef HIST_SMEM_NO_BANK_CONFLICT
}