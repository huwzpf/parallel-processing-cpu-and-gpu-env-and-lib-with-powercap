/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "cudampilib.h"
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

struct __cudampi__arguments_type __cudampi__arguments;

long long VECTORSIZE;

double *vectora;
double *vectorb;
double *vectorc;

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

  streamcount = __cudampi__arguments.number_of_streams;
  batchsize = __cudampi__arguments.batch_size;
  VECTORSIZE = VECADD_VECTORSIZE;

  assert(batchsize % VECADD_THREADS_IN_BLOCK == 0);

  int alldevicescount = 0;

  __cudampi__getDeviceCount(&alldevicescount);

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

  #pragma omp parallel num_threads(alldevicescount)
  {

    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devPtra, *devPtrb, *devPtrc;
    void *devPtra2, *devPtrb2, *devPtrc2;
    int i;
    cudaStream_t stream1;
    cudaStream_t stream2;
    int mythreadid = omp_get_thread_num();
    void *devPtr;
    void *devPtr2;
    int privatecounter = 0;

    __cudampi__setDevice(mythreadid);

    #pragma omp barrier
    
    __cudampi__malloc(&devPtra, batchsize * sizeof(double));
    if (!devPtra) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }
    __cudampi__malloc(&devPtrb, batchsize * sizeof(double));

    if (!devPtrb) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }

    __cudampi__malloc(&devPtrc, batchsize * sizeof(double));
    if (!devPtrc) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }

    __cudampi__malloc(&devPtr, 3 * sizeof(void *));
    if (!devPtr) 
    {
      log_message(LOG_ERROR, "\nNot enough memory.");
      exit(-1);
    }

    if(streamcount == 2)
    {
      __cudampi__malloc(&devPtra2, batchsize * sizeof(double));
      if (!devPtra2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }

      __cudampi__malloc(&devPtrb2, batchsize * sizeof(double));
      if (!devPtrb2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }

      __cudampi__malloc(&devPtrc2, batchsize * sizeof(double));
      if (!devPtrc2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }

      __cudampi__malloc(&devPtr2, 3 * sizeof(void *));
      if (!devPtr2) 
      {
        log_message(LOG_ERROR, "\nNot enough memory.");
        exit(-1);
      }
    }

    __cudampi__streamCreate(&stream1);
    __cudampi__memcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + 2 * sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    if(streamcount == 2)
    {
      __cudampi__streamCreate(&stream2);
      __cudampi__memcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
      __cudampi__memcpyAsync(devPtr2 + sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
      __cudampi__memcpyAsync(devPtr2 + 2 * sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
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
        __cudampi__memcpyAsync(devPtra, vectora + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream1);
        __cudampi__memcpyAsync(devPtrb, vectorb + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream1);
        __cudampi__kernelInStream(devPtr, stream1, 0);
        __cudampi__memcpyAsync(vectorc + batch_pointer.start, devPtrc, batch_pointer.n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream1);
        if (streamcount == 2) 
        {
            batch_pointer = __cudampi__getnextchunkindex(&globalcounter, VECTORSIZE);

            if (batch_pointer.start >= VECTORSIZE)
            {
              finish = 1;
            } 
            else 
            {
              __cudampi__memcpyAsync(devPtra2, vectora + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream2);
              __cudampi__memcpyAsync(devPtrb2, vectorb + batch_pointer.start, batch_pointer.n_elements * sizeof(double), cudaMemcpyHostToDevice, stream2);
              __cudampi__kernelInStream(devPtr2, stream2, 0);
              __cudampi__memcpyAsync(vectorc + batch_pointer.start, devPtrc2, batch_pointer.n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream2);
            }
        }
      }

      privatecounter++;
      if (privatecounter % 2 == 0) 
      {
        __cudampi__deviceSynchronize();
      }

    } while (!finish);

    __cudampi__deviceSynchronize();

    __cudampi__streamDestroy(stream1);
    __cudampi__free(devPtr);
    __cudampi__free(devPtra);
    __cudampi__free(devPtrb);
    __cudampi__free(devPtrc);
    if(streamcount == 2)
    {
      __cudampi__streamDestroy(stream2);
      __cudampi__free(devPtr2);
      __cudampi__free(devPtra2);
      __cudampi__free(devPtrb2);
      __cudampi__free(devPtrc2);
    }
  }

  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  __cudampi__terminateMPI();
  // save_vector_output_double(vectorc, VECTORSIZE, "vecadd_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vectora);
  cudaFreeHost(vectorb);
  cudaFreeHost(vectorc);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
