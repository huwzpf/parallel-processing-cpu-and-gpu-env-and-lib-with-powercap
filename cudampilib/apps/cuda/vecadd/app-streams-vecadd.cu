/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "vecadd_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"


long long VECTORSIZE;

double *vectora;
double *vectorb;
double *vectorc;

unsigned long batchsize;

long long globalcounter = 0;

int streamcount = 1;
float powerlimit = 0;


__global__ void appkernel(void *devPtr) 
{
  double *devPtra = (double *)(((void **)devPtr)[0]);
  double *devPtrb = (double *)(((void **)devPtr)[1]);
  double *devPtrc = (double *)(((void **)devPtr)[2]);

  long my_index = blockIdx.x * blockDim.x + threadIdx.x;
  devPtrc[my_index] = devPtra[my_index] / 2 + devPtrb[my_index] / 3;
}

void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream) 
{
  dim3 blocksingrid(batchSize / VECADD_THREADS_IN_BLOCK);
  dim3 threadsinblock(VECADD_THREADS_IN_BLOCK);

  log_message(LOG_DEBUG, "Launichng GPU Kernel with %i blocks in grid and %i threads in block.", batchSize / VECADD_THREADS_IN_BLOCK, VECADD_THREADS_IN_BLOCK);
  appkernel<<<blocksingrid, threadsinblock, 0, stream>>>(devPtr);

  if (cudaSuccess != cudaGetLastError()) {
    log_message(LOG_ERROR, "Error during kernel launch in stream");
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
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  streamcount = 2;
  batchsize = 480000;
  VECTORSIZE = 480000 * 100;

  assert(batchsize % VECADD_THREADS_IN_BLOCK == 0);

  cudaHostAlloc((void **)&vectora, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectora) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(-1);
  }

  cudaHostAlloc((void **)&vectorb, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorb) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(-1);
  }

  // Fill vectora with all 1's
  for (size_t i = 0; i < VECTORSIZE; i++) 
  {
      vectora[i] = ((int)i % 100);
      vectorb[i] = 1.0;
  }

  cudaHostAlloc((void **)&vectorc, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorc) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(-1);
  }

  gettimeofday(&start, NULL);


    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devPtra, *devPtrb, *devPtrc;
    void *devPtra2, *devPtrb2, *devPtrc2;
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
    cudaMalloc(&devPtrb, batchsize * sizeof(double));

    if (!devPtrb) 
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

    cudaMalloc(&devPtr, 3 * sizeof(void *));
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

    cudaMalloc(&devPtrb2, batchsize * sizeof(double));
    if (!devPtrb2) 
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

    cudaMalloc(&devPtr2, 3 * sizeof(void *));
    if (!devPtr2) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }

    cudaStreamCreate(&stream1);
    cudaMemcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync((char*)devPtr + sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync((char*)devPtr + 2 * sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);

    cudaStreamCreate(&stream2);
    cudaMemcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync((char*)devPtr2 + sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync((char*)devPtr2 + 2 * sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);

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
        cudaMemcpyAsync(devPtrb, vectorb + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream1);
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
          cudaMemcpyAsync(devPtrb2, vectorb + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream2);
          launchkernelinstream(devPtr, batchsize, stream2);
          cudaMemcpyAsync(vectorc + batch_pointer.start, devPtrc2, batch_pointer.n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream2);
        }

      }

    } while (!finish);

    cudaDeviceSynchronize();

    cudaStreamDestroy(stream1);
    cudaFree(devPtr);
    cudaFree(devPtra);
    cudaFree(devPtrb);
    cudaFree(devPtrc);

    cudaStreamDestroy(stream2);
    cudaFree(devPtr2);
    cudaFree(devPtra2);
    cudaFree(devPtrb2);
    cudaFree(devPtrc2);

  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  // save_vector_output_double(vectorc, VECTORSIZE, "vecadd_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vectora);
  cudaFreeHost(vectorb);
  cudaFreeHost(vectorc);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
