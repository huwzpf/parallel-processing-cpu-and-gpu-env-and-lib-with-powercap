/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef CUDAMPILIB_H
#define CUDAMPILIB_H

#include "cudampi.h"
#include <cuda_runtime.h>
#include <mpi.h>


typedef enum {
  DISABLED,
  CONTINOUS_EQUAL,
  BINARY_GREEDY,
  EDP_GRADIENT_SIMPLE,
  EDP_GRADIENT_SPSA
} powercapStrategy_t;

extern __global__ void kernel(long *devPtr);

struct __cudampi__arguments_type
{
  int cpu_enabled;
  int number_of_streams;
  unsigned long batch_size;
  float cpu_power_scaling;
  int cpu_batch_scaling_factor;
  int use_dynamic_scaling;
  float gpu_min_powercap;
  float cpu_min_powercap;
};

typedef struct
{
  long long start;
  unsigned long n_elements;
} __cudampi__batch_pointer;

void __cudampi__setglobalpowerlimit(float powerlimit);
int __cudampi__selectdevicesforpowerlimit_greedy();

__cudampi__batch_pointer __cudampi__getnextchunkindex(long long *globalcounter, long long max);
__cudampi__batch_pointer __cudampi__getnextchunkindex_enableddevices(long long *globalcounter, unsigned long batchsize, long long max);
__cudampi__batch_pointer __cudampi__getnextchunkindex_alldevices(long long *globalcounter, unsigned long batchsize, long long max);

unsigned long __cudampi__getCurrentBatchSize();

void __cudampi__initializeMPI(int argc, char **argv);

void __cudampi__terminateMPI();

int __cudampi__isCpu();

int __cudampi__gettargetGPU(int device);

int __cudampi__gettargetMPIrank(int device);

cudaError_t __cudampi__cudaMalloc(void **devPtr, size_t size);

cudaError_t __cudampi__cpuMalloc(void **devPtr, size_t size);

cudaError_t __cudampi__malloc(void **devPtr, size_t size);

cudaError_t __cudampi__cudaFree(void *devPtr);

cudaError_t __cudampi__cpuFree(void *devPtr);

cudaError_t __cudampi__free(void *devPtr);

cudaError_t __cudampi__deviceSynchronize(void);

cudaError_t __cudampi__cudaSetDevice(int device);

cudaError_t __cudampi__setDevice(int device);

cudaError_t __cudampi__cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

cudaError_t __cudampi__cpuMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

cudaError_t __cudampi__cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t __cudampi__cpuMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

void __cudampi__cudaKernel(void *devPtr, unsigned long batchsize, unsigned long long id);

void __cudampi__cudaKernelInStream(void *devPtr, unsigned long batchsize, cudaStream_t stream, unsigned long long id);

void __cudampi__cpuKernel(void *devPtr, unsigned long batchsize, unsigned long long id);

void __cudampi__cpuKernelInStream(void *devPtr, unsigned long batchsize, cudaStream_t stream, unsigned long long id);

cudaError_t __cudampi__cudaGetDeviceCount(int *count);

cudaError_t __cudampi__cpuGetDeviceCount(int *count);

cudaError_t  __cudampi__getDeviceCount (int *count);

cudaError_t __cudampi__cudaStreamCreate(cudaStream_t *pStream);

cudaError_t __cudampi__cudaStreamDestroy(cudaStream_t stream);

cudaError_t __cudampi__cpuStreamCreate(cudaStream_t *pStream);

cudaError_t __cudampi__cpuStreamDestroy(cudaStream_t stream);

cudaError_t __cudampi__streamCreate(cudaStream_t *stream);

cudaError_t __cudampi__streamDestroy(cudaStream_t stream);

cudaError_t __cudampi__memcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t __cudampi__memcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

void __cudampi__kernelInStream(void *devPtr, cudaStream_t stream, unsigned long long id);

void __cudampi__kernel(void *devPtr, unsigned long long id);

#endif
