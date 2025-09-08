/*
GPU side: simple 2D convolution over a batch of preprocessed images.
Input layout: N x C x H x W (row-major, contiguous per sample), float32.
Output: N x 1 x (H-K+1) x (W-K+1)
Weights: 1 x C x K x K
*/
#include <cuda.h>
#include <cuda_runtime.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "vision_defines.h"

// Access helpers
__device__ __forceinline__ int idx_in(int n, int c, int y, int x) {
  return (((n * VISION_IMG_C + c) * VISION_IMG_H) + y) * VISION_IMG_W + x;
}

__device__ __forceinline__ int idx_out(int n, int y, int x) {
  return ((n * VISION_OUT_H) + y) * VISION_OUT_W + x; // OUT_C = 1
}

// devPtr points to device memory containing 3 pointers: [input, weights, output]
__global__ void appkernel(void *devPtr) {
  float *in  = (float *)(((void **)devPtr)[0]);
  float *w   = (float *)(((void **)devPtr)[1]);
  float *out = (float *)(((void **)devPtr)[2]);

  const int OH = VISION_OUT_H;
  const int OW = VISION_OUT_W;

  long gid = blockIdx.x * blockDim.x + threadIdx.x;
  // long total = (long)gridDim.x * blockDim.x; // not needed but clarifies intent
  // long n_out_elems = (long)OH * OW * gridDim.y; // not used

  // We launch with total threads >= batchSize * OH * OW
  // Each thread computes one output pixel for one sample
  // Decode gid -> (n, y, x)

  // batch size is conveyed via launch geometry (gridDim.y or encoded via total threads)
  // We map flat gid across all outputs for all N: gid in [0, N*OH*OW)
  long N = (long)gridDim.y; // we pass N via grid.y in launcher
  long per_image = (long)OH * OW;
  long total_elems = N * per_image;
  if (gid >= total_elems) return;

  long n = gid / per_image;
  long rem = gid % per_image;
  int y = rem / OW;
  int x = rem % OW;

  // Single output channel
  float acc = 0.f;
  #pragma unroll
  for (int c = 0; c < VISION_IMG_C; ++c) {
    #pragma unroll
    for (int ky = 0; ky < VISION_K; ++ky) {
      #pragma unroll
      for (int kx = 0; kx < VISION_K; ++kx) {
        int iy = y + ky;
        int ix = x + kx;
        float val = in[idx_in((int)n, c, iy, ix)];
        // weights layout: [C, K, K]
        int widx = ((c * VISION_K) + ky) * VISION_K + kx;
        acc += val * w[widx];
      }
    }
  }

  out[idx_out((int)n, y, x)] = acc;
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long /* id */) {
  // Launch enough threads for batchSize * OH * OW elements
  const long OH = VISION_OUT_H;
  const long OW = VISION_OUT_W;
  long elems = (long)batchSize * OH * OW;

  int threads = VISION_THREADS_IN_BLOCK;
  int blocks = (int)((elems + threads - 1) / threads);

  // Encode batchSize into grid.y so kernel can derive N
  dim3 grid(blocks, (unsigned int)batchSize, 1);
  dim3 block(threads);

  log_message(LOG_DEBUG, "Launching GPU conv: N=%lu, OHxOW=%ldx%ld, blocks=%d, threads=%d", batchSize, OH, OW, blocks, threads);
  appkernel<<<grid, block, 0, stream>>>(devPtr);
  cudaError_t e = cudaGetLastError();
  if (cudaSuccess != e) {
    log_message(LOG_ERROR, "Error during kernel launch: %s", cudaGetErrorString(e));
  }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id) { launchkernelinstream(devPtr, batchSize, 0, id); }

