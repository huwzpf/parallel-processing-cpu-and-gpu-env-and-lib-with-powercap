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
#include <assert.h>

#include <sys/time.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "twinprime_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

struct __cudampi__arguments_type __cudampi__arguments;

long long VECTORSIZE;

long long *vector;
long long *results;

unsigned long batchsize;

long long globalcounter = 0;

int streamcount = 1;
float powerlimit = 0;

int main(int argc, char **argv) 
{

  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  streamcount = 2;
  batchsize = 480000;
  VECTORSIZE = TWINPRIME_PROBLEM_SIZE;

  assert(batchsize % TWINPRIME_THREADS_IN_BLOCK == 0);

  int alldevicescount = 0;

  __cudampi__getDeviceCount(&alldevicescount);

  log_message(LOG_INFO, "Malloc vector");

  cudaHostAlloc((void **)&vector, sizeof(long long) * VECTORSIZE, cudaHostAllocDefault);
  if (!vector) 
  {
    log_message(LOG_ERROR, "\nVector - not enough memory.");
    exit(-1);
  }

  cudaHostAlloc((void **)&results, sizeof(long long) * VECTORSIZE, cudaHostAllocDefault);
  if (!results) 
  {
    log_message(LOG_ERROR, "\nResults - not enough memory.");
    exit(-1);
  }

  log_message(LOG_INFO, "Malloc vector DONE %d", VECTORSIZE);

  // Filling input
  for (long long i = 0; i < VECTORSIZE; i++) {
    vector[i] = 1000000 + i;
  }

  gettimeofday(&start, NULL);

  #pragma omp parallel num_threads(alldevicescount)
  {

    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devVector = NULL;
    void *devVector2 = NULL;
    void *devResults = NULL;
    void *devResults2 = NULL;
    void *devPtr = NULL;
    void *devPtr2 = NULL;
    cudaStream_t stream;
    cudaStream_t stream2;
    long long privatecounter = 0;
    int mythreadid = omp_get_thread_num();

    cudaSetDevice(mythreadid);
    #pragma omp barrier

    cudaMalloc(&devVector, batchsize * sizeof(long long));
    if (!devVector) 
    {
      log_message(LOG_ERROR, "\ndevVector - not enough memory.");
      exit(-1);
    }

    cudaMalloc(&devResults, batchsize * sizeof(long long));
    if (!devResults) 
    {
      log_message(LOG_ERROR, "\ndevResults - not enough memory.");
      exit(-1);
    }

    cudaMalloc(&devPtr, 2 * sizeof(void *));
    if (!devPtr) 
    {
      log_message(LOG_ERROR, "\ndevPtr - not enough memory.");
      exit(-1);
    }

    if(streamcount == 2)
    {
      cudaMalloc(&devVector2, batchsize * sizeof(long long));
      if (!devVector2) 
      {
        log_message(LOG_ERROR, "\ndevVector - not enough memory.");
        exit(-1);
      }

      cudaMalloc(&devResults2, batchsize * sizeof(long long));
      if (!devResults2) 
      {
        log_message(LOG_ERROR, "\ndevResults - not enough memory.");
        exit(-1);
      }

      cudaMalloc(&devPtr2, 2 * sizeof(void *));
      if (!devPtr2) 
      {
        log_message(LOG_ERROR, "\ndevPtr - not enough memory.");
        exit(-1);
      }
    }

    cudaStreamCreate(&stream);
    cudaMemcpyAsync(devPtr, &devVector, sizeof(void *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(devPtr + sizeof(void *), &devResults, sizeof(void *), cudaMemcpyHostToDevice, stream);
    
    if (streamcount == 2) {
      cudaStreamCreate(&stream2);
      cudaMemcpyAsync(devPtr2, &devVector2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
      cudaMemcpyAsync(devPtr2 + sizeof(void *), &devResults2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    }
    do 
    {
      batch_pointer = __cudampi__getnextchunkindex(&globalcounter, VECTORSIZE);

      if (batch_pointer.start >= VECTORSIZE) 
      {
        finish = 1;
      } 
      else 
      {
        cudaMemcpyAsync(devVector, vector + batch_pointer.start, batch_pointer.n_elements * sizeof(long long), cudaMemcpyHostToDevice, stream);
        __cudampi__kernelInStream(devPtr, stream);
        cudaMemcpyAsync(results + batch_pointer.start, devResults, batch_pointer.n_elements * sizeof(long long), cudaMemcpyDeviceToHost, stream);

        if(streamcount == 2) 
        {
          batch_pointer = __cudampi__getnextchunkindex(&globalcounter, VECTORSIZE);

          if (batch_pointer.start >= VECTORSIZE) 
          {
            finish = 1;
          } 
          else 
          {
            cudaMemcpyAsync(devVector2, vector + batch_pointer.start, batch_pointer.n_elements * sizeof(long long), cudaMemcpyHostToDevice, stream2);
            __cudampi__kernelInStream(devPtr2, stream2);
            cudaMemcpyAsync(results + batch_pointer.start, devResults2, batch_pointer.n_elements * sizeof(long long), cudaMemcpyDeviceToHost, stream2);
          }
        }
      }

      privatecounter++;
      if (privatecounter % 2) 
      {
        cudaDeviceSynchronize();
      }
    } while (!finish);

    cudaStreamDestroy(stream);

    cudaFree(devPtr);
    cudaFree(devVector);
    cudaFree(devResults);

    if(streamcount == 2)
    {
      cudaStreamDestroy(stream2);
      cudaFree(devPtr2);
      cudaFree(devVector2);
      cudaFree(devResults2);
    }
  }
  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  __cudampi__terminateMPI();
  // save_vector_output_char(vectorc, VECTORSIZE, "patternsearch_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vector);
  cudaFreeHost(results);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
