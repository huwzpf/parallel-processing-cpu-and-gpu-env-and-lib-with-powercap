/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/time.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "collatz_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

long long VECTORSIZE;

double *vectora;
double *vectorc;

unsigned long batchsize;

long long globalcounter = 0;

float powerlimit;



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

void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream) 
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

typedef struct
{
  long long start;
  unsigned long n_elements;
} __cudampi__batch_pointer;

__cudampi__batch_pointer __cudampi__getnextchunkindex(long long *globalcounter, unsigned long batchsize, long long max) {
  // for a given thread (GPU) return the next available data chunk
  // max is the vector size
  __cudampi__batch_pointer batch_pointer = {max, 0};

    if(*globalcounter < max)
    {
      batch_pointer.start = *globalcounter;
      (*globalcounter) += batchsize;
    }


    if (batch_pointer.start < max)
    {
      batch_pointer.n_elements = (((batch_pointer.start + batchsize) > max )? max - batch_pointer.start : batchsize);
    }

  return batch_pointer;
}

int main(int argc, char **argv) 
{
  log_message(LOG_INFO, "Starting execution !");
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  batchsize = 480000;
  VECTORSIZE = 480000 * 100;

  assert(batchsize % COLLATZ_THREADS_IN_BLOCK == 0);

  cudaHostAlloc((void **)&vectora, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectora) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(-1);
  }

  cudaHostAlloc((void **)&vectorc, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorc) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(-1);
  }

  // Filling input
  for (long long i = 0; i < VECTORSIZE; i++) 
  {
    vectora[i] = (80000000 + i) % VECTORSIZE;
  }

  gettimeofday(&start, NULL);

    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devPtra, *devPtrc;
    void *devPtra2, *devPtrc2;
    cudaStream_t stream1;
    cudaStream_t stream2;
    void *devPtr;
    void *devPtr2;
    cudaSetDevice(0);
    cudaMalloc(&devPtra, batchsize * sizeof(double));
    if (!devPtra) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }
    cudaMalloc(&devPtrc, batchsize * sizeof(double));
    if (!devPtrc) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }

    cudaMalloc(&devPtr, 2 * sizeof(void *));
    if (!devPtr) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }


      cudaMalloc(&devPtra2, batchsize * sizeof(double));
      if (!devPtra2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }
      cudaMalloc(&devPtrc2, batchsize * sizeof(double));
      if (!devPtrc2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }

      cudaMalloc(&devPtr2, 2 * sizeof(void *));
      if (!devPtr2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }


    cudaStreamCreate(&stream1);
    cudaMemcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(((char*)devPtr + sizeof(void *)), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);

    cudaStreamCreate(&stream2);
    cudaMemcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(((char*)devPtr2 + sizeof(void *)), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);

    do 
    {
      batch_pointer = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);

      if (batch_pointer.start >= VECTORSIZE) 
      {
        finish = 1;
      }
      else 
      {
        cudaMemcpyAsync(devPtra, vectora + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream1);
        launchkernelinstream(devPtr, batchsize, stream1);
        cudaMemcpyAsync(vectorc + batch_pointer.start, devPtrc, batch_pointer.n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream1);

          batch_pointer = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);

          if (batch_pointer.start >= VECTORSIZE) 
          {
            finish = 1;
          } 
          else 
          {
            cudaMemcpyAsync(devPtra2, vectora + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream2);
            launchkernelinstream(devPtr, batchsize, stream1);
            cudaMemcpyAsync(vectorc + batch_pointer.start, devPtrc2, batch_pointer.n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream2);
          }

      }
    } while (!finish);

    cudaDeviceSynchronize();

    cudaStreamDestroy(stream1);
    cudaFree(devPtr);
    cudaFree(devPtra);
    cudaFree(devPtrc);
      cudaStreamDestroy(stream2);
      cudaFree(devPtr2);
      cudaFree(devPtra2);
      cudaFree(devPtrc2);

  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  // save_vector_output_double(vectorc, VECTORSIZE, "collatz_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vectora);
  cudaFreeHost(vectorc);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}