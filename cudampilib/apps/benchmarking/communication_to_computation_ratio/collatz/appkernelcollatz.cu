/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
// idea: in most cases, input data can be uploaded to GPU's memory and
// consequently we only need to copy a pointer in kernel invocation
// in OpenCL we could hide any kernel invocation

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "collatz_defines.h"

__device__ int isprime(long a) 
{
  long i;
  for (i = 2; i < sqrt((double)a) + 1; i++) 
  {
    if ((a % i) == 0) 
    {
      return 0;
    }
  }
  return 1;
}

__global__ void appkernel(void *devPtr) 
{
  double *devPtra = (double *)(((void **)devPtr)[0]);
  double *devPtrc = (double *)(((void **)devPtr)[1]);

  long my_index = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned long start = devPtra[my_index];
  unsigned long counter = 0;

  if (isprime(start)) 
  {
    for (; (start > 1); counter++) 
    {
      start = (start % 2) ? (3 * start + 1) : (start / 2);
    }
  }

  devPtrc[my_index] = counter;
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long id) 
{
  // BLOCKS_IN_GRID = batch_size / 64
  dim3 blocksingrid(batchSize / COLLATZ_THREADS_IN_BLOCK);
  dim3 threadsinblock(COLLATZ_THREADS_IN_BLOCK);

  log_message(LOG_DEBUG, "Launichng GPU Kernel with %i blocks in grid and %i threads in block.", batchSize / COLLATZ_THREADS_IN_BLOCK, COLLATZ_THREADS_IN_BLOCK);
  appkernel<<<blocksingrid, threadsinblock, 0, stream>>>(devPtr);

  cudaError_t e = cudaGetLastError();
  if (cudaSuccess != e) {
    log_message(LOG_ERROR, "Error during kernel launch in stream, %s", cudaGetErrorString(e));
  }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id) { launchkernelinstream(devPtr, batchSize, 0, id); }
