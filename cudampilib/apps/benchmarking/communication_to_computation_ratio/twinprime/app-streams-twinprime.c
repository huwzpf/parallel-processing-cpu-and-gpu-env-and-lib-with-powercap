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

float powerlimit = 0;

double total_communication_time = 0.0;
double total_computation_time = 0.0;

int main(int argc, char **argv) 
{

  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  batchsize = __cudampi__arguments.batch_size;
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

    double time_memcpy_d2h = 0.0;
    double time_memcpy_h2d = 0.0;
    double time_kernel = 0.0;
    double time_synchronize = 0.0;
    struct timeval start_memcpy_d2h, stop_memcpy_d2h;
    struct timeval start_memcpy_h2d, stop_memcpy_h2d;
    struct timeval start_kernel, stop_kernel;
    struct timeval start_synchronize, stop_synchronize;
  
    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devVector = NULL;
    void *devResults = NULL;
    void *devPtr = NULL;
    long long privatecounter = 0;
    int mythreadid = omp_get_thread_num();

    __cudampi__setDevice(mythreadid);
    #pragma omp barrier

    __cudampi__malloc(&devVector, batchsize * sizeof(long long));
    if (!devVector) 
    {
      log_message(LOG_ERROR, "\ndevVector - not enough memory.");
      exit(-1);
    }

    __cudampi__malloc(&devResults, batchsize * sizeof(long long));
    if (!devResults) 
    {
      log_message(LOG_ERROR, "\ndevResults - not enough memory.");
      exit(-1);
    }

    __cudampi__malloc(&devPtr, 2 * sizeof(void *));
    if (!devPtr) 
    {
      log_message(LOG_ERROR, "\ndevPtr - not enough memory.");
      exit(-1);
    }

    gettimeofday(&start_memcpy_h2d, NULL);
    __cudampi__memcpy(devPtr, &devVector, sizeof(void *), cudaMemcpyHostToDevice);
    __cudampi__memcpy(devPtr + sizeof(void *), &devResults, sizeof(void *), cudaMemcpyHostToDevice);
    gettimeofday(&stop_memcpy_h2d, NULL);
    time_memcpy_h2d += (double)((stop_memcpy_h2d.tv_sec - start_memcpy_h2d.tv_sec) + (double)(stop_memcpy_h2d.tv_usec - start_memcpy_h2d.tv_usec) / 1000000.0);
        
    do 
    {
      batch_pointer = __cudampi__getnextchunkindex(&globalcounter, VECTORSIZE);

      if (batch_pointer.start >= VECTORSIZE) 
      {
        finish = 1;
      } 
      else 
      {
        gettimeofday(&start_memcpy_h2d, NULL);
        __cudampi__memcpy(devVector, vector + batch_pointer.start, batch_pointer.n_elements * sizeof(long long), cudaMemcpyHostToDevice);
        gettimeofday(&stop_memcpy_h2d, NULL);
        time_memcpy_h2d += (double)((stop_memcpy_h2d.tv_sec - start_memcpy_h2d.tv_sec) + (double)(stop_memcpy_h2d.tv_usec - start_memcpy_h2d.tv_usec) / 1000000.0);
        
        gettimeofday(&start_kernel, NULL);
        __cudampi__kernel(devPtr);
        gettimeofday(&stop_kernel, NULL);
        time_kernel += (double)((stop_kernel.tv_sec - start_kernel.tv_sec) + (double)(stop_kernel.tv_usec - start_kernel.tv_usec) / 1000000.0);
        
        gettimeofday(&start_synchronize, NULL);
        __cudampi__deviceSynchronize();
        gettimeofday(&stop_synchronize, NULL);
        time_synchronize += (double)((stop_synchronize.tv_sec - start_synchronize.tv_sec) + (double)(stop_synchronize.tv_usec - start_synchronize.tv_usec) / 1000000.0);
      
        gettimeofday(&start_memcpy_d2h, NULL);
        __cudampi__memcpy(results + batch_pointer.start, devResults, batch_pointer.n_elements * sizeof(long long), cudaMemcpyDeviceToHost);
        gettimeofday(&stop_memcpy_d2h, NULL);
        time_memcpy_d2h += (double)((stop_memcpy_d2h.tv_sec - start_memcpy_d2h.tv_sec) + (double)(stop_memcpy_d2h.tv_usec - start_memcpy_d2h.tv_usec) / 1000000.0);
      
      }
    } while (!finish);

    gettimeofday(&start_synchronize, NULL);
    __cudampi__deviceSynchronize();
    gettimeofday(&stop_synchronize, NULL);
    time_synchronize += (double)((stop_synchronize.tv_sec - start_synchronize.tv_sec) + (double)(stop_synchronize.tv_usec - start_synchronize.tv_usec) / 1000000.0);



    __cudampi__free(devPtr);
    __cudampi__free(devVector);
    __cudampi__free(devResults);
  
    log_message(LOG_INFO, "Memcpy H2D time: %f s, Memcpy D2H time: %f s, Kernel time: %f s, Synchronize time: %f s\n",
           mythreadid, time_memcpy_h2d, time_memcpy_d2h, time_kernel, time_synchronize);

    #pragma omp atomic
    total_communication_time += time_memcpy_h2d + time_memcpy_d2h;
    #pragma omp atomic
    total_computation_time += time_kernel + time_synchronize;

  }
  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));
  log_message(LOG_INFO, "Total communication time=%f s, Total computation time=%f s\n", total_communication_time, total_computation_time);

  __cudampi__terminateMPI();
  // save_vector_output_char(vectorc, VECTORSIZE, "patternsearch_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vector);
  cudaFreeHost(results);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
