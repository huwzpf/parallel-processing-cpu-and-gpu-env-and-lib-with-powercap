/*
Copyright 2025
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "montecarlo_defines.h"

// Simple LCG for deterministic per-thread RNG (avoid extra library deps)
__device__ inline unsigned int lcg_next(unsigned int x) {
  return 1664525u * x + 1013904223u; // Numerical Recipes LCG
}

__device__ inline float u01_from_uint(unsigned int x) {
  // Convert to (0,1) float
  return (x >> 8) * (1.0f / 16777216.0f); // 24-bit mantissa
}

__global__ void appkernel(void *devPtr)
{
  // devPtr holds two pointers: seeds, hits
  unsigned int *seeds = (unsigned int*)(((void**)devPtr)[0]);
  unsigned int *hits  = (unsigned int*)(((void**)devPtr)[1]);

  unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int state = seeds[idx] ^ (unsigned int)idx;
  unsigned int local_hits = 0u;

  #pragma unroll 4
  for (int i = 0; i < MONTECARLO_ITERS_PER_ITEM; ++i) {
    state = lcg_next(state);
    float x = 2.0f * u01_from_uint(state) - 1.0f;
    state = lcg_next(state);
    float y = 2.0f * u01_from_uint(state) - 1.0f;
    float r2 = x * x + y * y;
    local_hits += (r2 <= 1.0f);
  }

  hits[idx] = local_hits;
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long /* id */)
{
  dim3 blocks((unsigned int)(batchSize / MONTECARLO_THREADS_IN_BLOCK));
  dim3 threads(MONTECARLO_THREADS_IN_BLOCK);

  log_message(LOG_DEBUG, "Launching Monte Carlo GPU kernel with %u blocks x %u threads.",
              (unsigned)blocks.x, (unsigned)threads.x);
  appkernel<<<blocks, threads, 0, stream>>>(devPtr);

  cudaError_t e = cudaGetLastError();
  if (cudaSuccess != e) {
    log_message(LOG_ERROR, "Error during Monte Carlo kernel launch: %s", cudaGetErrorString(e));
  }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id) {
  launchkernelinstream(devPtr, batchSize, 0, id);
}

