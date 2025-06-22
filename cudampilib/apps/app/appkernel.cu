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

#include "logger_gpu.h"
#define ENABLE_LOGGING_GPU

__global__ void appkernel(unsigned char *devPtr) {
  long my_index = blockIdx.x * blockDim.x + threadIdx.x;

  devPtr[my_index] = devPtr[my_index] / 2;
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long id) {

  dim3 blocksingrid(2);
  dim3 threadsinblock(1024);

  appkernel<<<blocksingrid, threadsinblock, 0, stream>>>((unsigned char *)devPtr);

  if (cudaSuccess != cudaGetLastError()) {
    log_message_gpu(LOG_ERROR_GPU, "Error during kernel launch in stream");
  }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id) { launchkernelinstream(devPtr, batchSize, 0, id); }
