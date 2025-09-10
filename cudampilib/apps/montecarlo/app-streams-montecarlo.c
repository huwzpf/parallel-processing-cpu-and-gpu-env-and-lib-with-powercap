/*
Copyright 2025
*/
#include "cudampilib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "montecarlo_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

struct __cudampi__arguments_type __cudampi__arguments;

long long VECTORSIZE;

unsigned int *seeds;
unsigned int *hits;

unsigned long batchsize;

long long globalcounter = 0;

int streamcount = 1;

// As in the RNN app: simulate a larger problem by iterating over the
// same allocated buffers multiple times to avoid huge host memory usage.
#define ITERS 1

int main(int argc, char **argv)
{
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  streamcount = __cudampi__arguments.number_of_streams;
  batchsize   = __cudampi__arguments.batch_size;
  VECTORSIZE  = MONTECARLO_PROBLEM_SIZE;

  assert(batchsize % MONTECARLO_THREADS_IN_BLOCK == 0);

  int alldevicescount = 0;
  __cudampi__getDeviceCount(&alldevicescount);

  cudaHostAlloc((void **)&seeds, sizeof(unsigned int) * VECTORSIZE, cudaHostAllocDefault);
  if (!seeds) { log_message(LOG_ERROR, "Not enough memory for seeds."); exit(-1); }
  cudaHostAlloc((void **)&hits,  sizeof(unsigned int) * VECTORSIZE, cudaHostAllocDefault);
  if (!hits)  { log_message(LOG_ERROR, "Not enough memory for hits."); exit(-1); }

  // Fill seeds deterministically
  for (long long i = 0; i < VECTORSIZE; ++i) {
    seeds[i] = (unsigned int)(i * 2654435761u + 12345u);
  }

  gettimeofday(&start, NULL);

  long long total_sync_intervals = 0;
  long long total_sync_sum_us = 0;

  #pragma omp parallel num_threads(alldevicescount) reduction(+:total_sync_intervals,total_sync_sum_us)
  {
    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devSeeds = NULL, *devHits = NULL;
    void *devPtr = NULL;
    void *devSeeds2 = NULL, *devHits2 = NULL, *devPtr2 = NULL;
    cudaStream_t stream1, stream2;
    long long privatecounter = 0;
    int mythreadid = omp_get_thread_num();

    __cudampi__setDevice(mythreadid);
    #pragma omp barrier

    __cudampi__malloc(&devSeeds, batchsize * sizeof(unsigned int));
    if (!devSeeds) { log_message(LOG_ERROR, "devSeeds OOM"); exit(-1); }
    __cudampi__malloc(&devHits,  batchsize * sizeof(unsigned int));
    if (!devHits)  { log_message(LOG_ERROR, "devHits OOM"); exit(-1); }

    __cudampi__malloc(&devPtr, 2 * sizeof(void*));
    if (!devPtr)   { log_message(LOG_ERROR, "devPtr OOM"); exit(-1); }

    if (streamcount == 2) {
      __cudampi__malloc(&devSeeds2, batchsize * sizeof(unsigned int));
      if (!devSeeds2) { log_message(LOG_ERROR, "devSeeds2 OOM"); exit(-1); }
      __cudampi__malloc(&devHits2,  batchsize * sizeof(unsigned int));
      if (!devHits2)  { log_message(LOG_ERROR, "devHits2 OOM"); exit(-1); }
      __cudampi__malloc(&devPtr2, 2 * sizeof(void*));
      if (!devPtr2)   { log_message(LOG_ERROR, "devPtr2 OOM"); exit(-1); }
    }

    __cudampi__streamCreate(&stream1);
    __cudampi__memcpyAsync(devPtr, &devSeeds, sizeof(void*), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + sizeof(void*), &devHits, sizeof(void*), cudaMemcpyHostToDevice, stream1);
    if (streamcount == 2) {
      __cudampi__streamCreate(&stream2);
      __cudampi__memcpyAsync(devPtr2, &devSeeds2, sizeof(void*), cudaMemcpyHostToDevice, stream2);
      __cudampi__memcpyAsync(devPtr2 + sizeof(void*), &devHits2, sizeof(void*), cudaMemcpyHostToDevice, stream2);
    }

    struct timeval last_sync_time; int has_last_sync_time = 0;
    long long local_sync_intervals = 0, local_sync_sum_us = 0;

    do {
      batch_pointer = __cudampi__getnextchunkindex(&globalcounter, ITERS * VECTORSIZE);
      if (batch_pointer.start >= ITERS * VECTORSIZE) {
        finish = 1;
      } else {
        // Simulate larger memory by counting up to ITERS*VECTORSIZE while only
        // VECTORSIZE elements are allocated. Keep batch_pointer within bounds.
        // (VECTORSIZE - batchsize) is the largest safe start (n_elements <= batchsize).
        batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
        __cudampi__memcpyAsync(devSeeds, seeds + batch_pointer.start, batch_pointer.n_elements * sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
        __cudampi__kernelInStream(devPtr, stream1, 0);
        __cudampi__memcpyAsync(hits + batch_pointer.start, devHits, batch_pointer.n_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1);

        if (streamcount == 2) {
          batch_pointer = __cudampi__getnextchunkindex(&globalcounter, ITERS * VECTORSIZE);
          if (batch_pointer.start >= ITERS * VECTORSIZE) {
            finish = 1;
          } else {
            batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
            __cudampi__memcpyAsync(devSeeds2, seeds + batch_pointer.start, batch_pointer.n_elements * sizeof(unsigned int), cudaMemcpyHostToDevice, stream2);
            __cudampi__kernelInStream(devPtr2, stream2, 0);
            __cudampi__memcpyAsync(hits + batch_pointer.start, devHits2, batch_pointer.n_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream2);
          }
        }
      }

      privatecounter++;
      if (privatecounter % 10 == 0) {
        __cudampi__deviceSynchronize();
        struct timeval now_sync; gettimeofday(&now_sync, NULL);
        if (has_last_sync_time) {
          long long delta_us = (now_sync.tv_sec - last_sync_time.tv_sec) * 1000000LL + (now_sync.tv_usec - last_sync_time.tv_usec);
          local_sync_sum_us += delta_us; local_sync_intervals += 1;
        }
        last_sync_time = now_sync; has_last_sync_time = 1;
      }
    } while (!finish);

    total_sync_intervals += local_sync_intervals;
    total_sync_sum_us += local_sync_sum_us;

    __cudampi__streamDestroy(stream1);
    __cudampi__free(devPtr);
    __cudampi__free(devSeeds);
    __cudampi__free(devHits);
    if (streamcount == 2) {
      __cudampi__streamDestroy(stream2);
      __cudampi__free(devPtr2);
      __cudampi__free(devSeeds2);
      __cudampi__free(devHits2);
    }
  }

  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  if (total_sync_intervals > 0) {
    double avg_us = (double) total_sync_sum_us / (double) total_sync_intervals;
    log_message(LOG_INFO, "Average period between deviceSynchronize() calls: %.3f ms over %lld intervals\n", avg_us / 1000.0, total_sync_intervals);
  } else {
    log_message(LOG_INFO, "No deviceSynchronize() pairs observed inside main loop.\n");
  }

  __cudampi__terminateMPI();

  // Optionally compute Pi estimate summary
  // Avoid heavy I/O; compute simple aggregate and print on rank 0
  unsigned long long total_samples = (unsigned long long)VECTORSIZE * (unsigned long long)MONTECARLO_ITERS_PER_ITEM;
  unsigned long long inside = 0ULL;
  for (long long i = 0; i < VECTORSIZE; ++i) inside += hits[i];
  double pi_est = 4.0 * ((double)inside / (double)total_samples);
  log_message(LOG_INFO, "Pi estimate: %.6f (samples=%llu)", pi_est, total_samples);

  cudaFreeHost(seeds);
  cudaFreeHost(hits);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
