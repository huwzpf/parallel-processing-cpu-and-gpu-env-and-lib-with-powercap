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
#define ITERS 50

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

int streamcount = 1;
float powerlimit;

int main(int argc, char **argv) 
{
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  streamcount = __cudampi__arguments.number_of_streams;
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

  // Measure average period between __cudampi__deviceSynchronize() calls in the main loop
  long long total_sync_intervals = 0;      // aggregated across threads
  long long total_sync_sum_us = 0;         // aggregated across threads (microseconds)

  #pragma omp parallel num_threads(alldevicescount) reduction(+:total_sync_intervals,total_sync_sum_us)
  {
    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devPtra, *devPtrc;
    void *devPtra2, *devPtrc2;
    void *devPtrk ,*devPtrk2;
    void *devPtrb ,*devPtrb2;
    int i;
    cudaStream_t stream1;
    cudaStream_t stream2;
    int mythreadid = omp_get_thread_num();
    void *devPtr;
    void *devPtr2;
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

    if(streamcount == 2)
    {
      __cudampi__malloc(&devPtra2, INPUT_BATCH_SIZE * batchsize * sizeof(double));
      if (!devPtra2) 
      {
        log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
        exit(-1);
      }
      
      __cudampi__malloc(&devPtrc2, OUTPUT_BATCH_SIZE * batchsize * sizeof(double));
      if (!devPtrc2) 
      {
        log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
        exit(-1);
      }
      __cudampi__malloc(&devPtrk2, WEIGHTS_SIZE * sizeof(double));
      if (!devPtrk2) 
      {
        log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
        exit(-1);
      }
      __cudampi__malloc(&devPtrb2, batchsize * HIDDEN_BATCH_SIZE * sizeof(double));
      if (!devPtrb2) 
      {
        log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
        exit(-1);
      }
      __cudampi__malloc(&devPtr2, 4 * sizeof(void *));
      if (!devPtr2) 
      {
        log_message(LOG_ERROR, "[Thread %d] Not enough memory.", omp_get_thread_num());
        exit(-1);
      }

    }
    __cudampi__streamCreate(&stream1);
    __cudampi__memcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + 2 * sizeof(void *), &devPtrk, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + 3 * sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtrk, vectork, WEIGHTS_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream1);
    if (streamcount == 2)
    {
    __cudampi__streamCreate(&stream2);
    __cudampi__memcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__memcpyAsync(devPtr2 + sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__memcpyAsync(devPtr2 + 2 * sizeof(void *), &devPtrk2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__memcpyAsync(devPtr2 + 3 * sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__memcpyAsync(devPtrk2, vectork, WEIGHTS_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream2);
    }
    // Local trackers for sync call periods (per-thread)
    struct timeval last_sync_time;
    int has_last_sync_time = 0;
    long long local_sync_intervals = 0;
    long long local_sync_sum_us = 0;

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

        __cudampi__memcpyAsync(devPtra, vectora + (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream1);
        __cudampi__kernelInStream(devPtr, stream1, 0);
        __cudampi__memcpyAsync(vectorc + (batch_pointer.start * OUTPUT_BATCH_SIZE), devPtrc, OUTPUT_BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream1);

        if (streamcount == 2) 
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
            //log_message(LOG_INFO, "[Thread %d] Sending chunk %ld with elements %ld (%ld , %ld), devPtr=%lld", omp_get_thread_num(), batch_pointer.start, batch_pointer.n_elements, (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE, devPtr2);


            __cudampi__memcpyAsync(devPtra2, vectora + (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream2);
            __cudampi__kernelInStream(devPtr2, stream2, 0);
            __cudampi__memcpyAsync(vectorc + (batch_pointer.start * OUTPUT_BATCH_SIZE), devPtrc2,  OUTPUT_BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream2);
          }
        }
      }

      //log_message(LOG_INFO, "[Thread %d] Completed iteration %d", omp_get_thread_num(), privatecounter);
      __cudampi__deviceSynchronize();

      // Record period between consecutive deviceSynchronize() calls inside the main loop
      struct timeval now_sync;
      gettimeofday(&now_sync, NULL);
      if (has_last_sync_time)
      {
        long long delta_us = (now_sync.tv_sec - last_sync_time.tv_sec) * 1000000LL +
                              (now_sync.tv_usec - last_sync_time.tv_usec);
        local_sync_sum_us += delta_us;
        local_sync_intervals += 1;
      }
      last_sync_time = now_sync;
      has_last_sync_time = 1;

    } while (!finish);

    __cudampi__deviceSynchronize();

    // Contribute local stats to the reduction totals
    total_sync_intervals += local_sync_intervals;
    total_sync_sum_us += local_sync_sum_us;

    __cudampi__streamDestroy(stream1);
    __cudampi__free(devPtr);
    __cudampi__free(devPtra);
    __cudampi__free(devPtrc);
    __cudampi__free(devPtrk);
    __cudampi__free(devPtrb);
    if(streamcount == 2)
    {
      __cudampi__streamDestroy(stream2);
      __cudampi__free(devPtr2);
      __cudampi__free(devPtra2);
      __cudampi__free(devPtrc2);
      __cudampi__free(devPtrk2);
      __cudampi__free(devPtrb2);
    }
  }
  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  if (total_sync_intervals > 0)
  {
    double avg_us = (double) total_sync_sum_us / (double) total_sync_intervals;
    log_message(LOG_INFO, "Average period between deviceSynchronize() calls: %.3f ms over %lld intervals\n",
                avg_us / 1000.0, total_sync_intervals);
  }
  else
  {
    log_message(LOG_INFO, "No deviceSynchronize() pairs observed inside main loop.\n");
  }

  __cudampi__terminateMPI();
  // save_vector_output_double(vectorc, VECTORSIZE, "RNN_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vectora);
  cudaFreeHost(vectorc);
  cudaFreeHost(vectork);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
