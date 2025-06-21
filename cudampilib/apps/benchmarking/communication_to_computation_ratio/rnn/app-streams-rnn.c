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
#include<unistd.h>

#include <sys/time.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "rnn_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

// For VECTORSIZE = 20000 this application uses 24.5 GB of host memory
// Just to hold input and output data
// By scaling it 30x, we get accurate benchmark results that would need 250 GB
// Another problem might be host memory used by MPI for sending data around
// So looks like it's better to just execute multiple iterations on the same data
// And don't risk running out of memory
#define ITERS 5

struct __cudampi__arguments_type __cudampi__arguments;

long long VECTORSIZE;

double *vectora;
double *vectorc;
double *vectork;
double *W_hh;
double *W_ih;
double *W_ho;

unsigned long batchsize;

long long globalcounter = 0;

float powerlimit;
double total_communication_time = 0.0;
double total_computation_time = 0.0;

int main(int argc, char **argv) 
{
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);
  batchsize = __cudampi__arguments.batch_size;
  VECTORSIZE = RNN_VECTORSIZE;

  assert(RNN_HIDDEN_SIZE >= RNN_INPUT_SIZE && RNN_HIDDEN_SIZE >= RNN_OUTPUT_SIZE);

  int alldevicescount = 0;

  __cudampi__getDeviceCount(&alldevicescount);

  cudaHostAlloc((void **)&vectora, sizeof(double) * VECTORSIZE * INPUT_BATCH_SIZE, cudaHostAllocDefault);
  if (!vectora) 
  {
    log_message(LOG_ERROR, "\nNot enough memory for vectora.");
    exit(-1);
  }

  cudaHostAlloc((void **)&vectorc, sizeof(double) * VECTORSIZE * OUTPUT_BATCH_SIZE, cudaHostAllocDefault);
  if (!vectorc) 
  {
    log_message(LOG_ERROR, "\nNot enough memory for vectorc.");
    exit(-1);
  }
  cudaHostAlloc((void **)&vectork, sizeof(double) * WEIGHTS_SIZE, cudaHostAllocDefault);
  if (!vectork) 
  {
    log_message(LOG_ERROR, "\nNot enough memory for vectork.");
    exit(-1);
  }

  for (long long i = 0; i < WEIGHTS_SIZE; i++) 
  {
    vectork[i] = (((double)rand())/((double)RAND_MAX));
  }

  for (long long i = 0; i < (VECTORSIZE * INPUT_BATCH_SIZE); i++) 
  {
    vectora[i] = i + vectork[i % WEIGHTS_SIZE];
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
    void *devPtra, *devPtrc;
    void *devPtrk;
    void *devPtrb;
    int i;
    int mythreadid = omp_get_thread_num();
    void *devPtr;
    void *devPtr2;
    long long privatecounter = 0;
    __cudampi__setDevice(mythreadid);
    #pragma omp barrier
    
    __cudampi__malloc(&devPtra, INPUT_BATCH_SIZE * batchsize * sizeof(double));
    if (!devPtra) 
    {
      log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
      exit(-1);
    }
    
    __cudampi__malloc(&devPtrc, OUTPUT_BATCH_SIZE * batchsize * sizeof(double));
    if (!devPtrc) 
    {
      log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
      exit(-1);
    }
    
    __cudampi__malloc(&devPtrk, WEIGHTS_SIZE * sizeof(double));
    if (!devPtrk) 
    {
      log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
      exit(-1);
    }
    
    __cudampi__malloc(&devPtrb, batchsize * HIDDEN_BATCH_SIZE * sizeof(double));
    if (!devPtrb) 
    {
      log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
      exit(-1);
    }
    
    __cudampi__malloc(&devPtr, 4 * sizeof(void *));
    if (!devPtr) 
    {
      log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
      exit(-1);
    }

    gettimeofday(&start_memcpy_h2d, NULL);
    __cudampi__memcpy(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice);
    __cudampi__memcpy(devPtr + sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice);
    __cudampi__memcpy(devPtr + 2 * sizeof(void *), &devPtrk, sizeof(void *), cudaMemcpyHostToDevice);
    __cudampi__memcpy(devPtr + 3 * sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice);
    __cudampi__memcpy(devPtrk, vectork, WEIGHTS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    gettimeofday(&stop_memcpy_h2d, NULL);
    time_memcpy_h2d += (double)((stop_memcpy_h2d.tv_sec - start_memcpy_h2d.tv_sec) + (double)(stop_memcpy_h2d.tv_usec - start_memcpy_h2d.tv_usec) / 1000000.0);
    
    do 
    {
      batch_pointer = __cudampi__getnextchunkindex(&globalcounter, ITERS * VECTORSIZE);

      if (batch_pointer.start >= ITERS * VECTORSIZE) 
      {
        finish = 1;
      }
      else 
      {
        // "Simulate" larger memory size by counting all the way to ITERS * VECTORSIZE (while only VECTORSIZE will fit into RAM)
        // (VECTORSIZE - batchsize) is largest value that batch_pointer.start can safely take (as n_elements <= batchsize)
        batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
        //log_message(LOG_INFO, "[Thread %d] Sending chunk %ld with elements %ld (%ld , %ld), devPtr=%lld", omp_get_thread_num(), batch_pointer.start, batch_pointer.n_elements, (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE, devPtr);

        gettimeofday(&start_memcpy_h2d, NULL);
        __cudampi__memcpy(devPtra, vectora + (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(double), cudaMemcpyHostToDevice);
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
        __cudampi__memcpy(vectorc + (batch_pointer.start * OUTPUT_BATCH_SIZE), devPtrc, OUTPUT_BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        gettimeofday(&stop_memcpy_d2h, NULL);
        time_memcpy_d2h += (double)((stop_memcpy_d2h.tv_sec - start_memcpy_d2h.tv_sec) + (double)(stop_memcpy_d2h.tv_usec - start_memcpy_d2h.tv_usec) / 1000000.0);
      }

    } while (!finish);

    gettimeofday(&start_synchronize, NULL);
    __cudampi__deviceSynchronize();
    gettimeofday(&stop_synchronize, NULL);
    time_synchronize += (double)((stop_synchronize.tv_sec - start_synchronize.tv_sec) + (double)(stop_synchronize.tv_usec - start_synchronize.tv_usec) / 1000000.0);

    __cudampi__free(devPtr);
    __cudampi__free(devPtra);
    __cudampi__free(devPtrc);
    __cudampi__free(devPtrk);
    __cudampi__free(devPtrb);
    
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
  // save_vector_output_double(vectorc, VECTORSIZE, "RNN_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vectora);
  cudaFreeHost(vectorc);
  cudaFreeHost(vectork);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}