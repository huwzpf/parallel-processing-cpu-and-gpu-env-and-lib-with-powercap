/*
Copyright 2025
*/
#include <omp.h>
#include <math.h>
#include <stdio.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "montecarlo_defines.h"

static inline unsigned int lcg_next(unsigned int x) {
  return 1664525u * x + 1013904223u;
}

static inline float u01_from_uint(unsigned int x) {
  return (x >> 8) * (1.0f / 16777216.0f);
}

static void appkernel(void *devPtr, unsigned long num_elements, int num_threads)
{
  unsigned int *seeds = (unsigned int*)(((void**)devPtr)[0]);
  unsigned int *hits  = (unsigned int*)(((void**)devPtr)[1]);

  #pragma omp parallel for num_threads(num_threads)
  for (unsigned long i = 0; i < num_elements; ++i) {
    unsigned int state = seeds[i] ^ (unsigned int)i;
    unsigned int local_hits = 0u;
    for (int k = 0; k < MONTECARLO_ITERS_PER_ITEM; ++k) {
      state = lcg_next(state);
      float x = 2.0f * u01_from_uint(state) - 1.0f;
      state = lcg_next(state);
      float y = 2.0f * u01_from_uint(state) - 1.0f;
      float r2 = x * x + y * y;
      local_hits += (r2 <= 1.0f);
    }
    hits[i] = local_hits;
  }
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long /* id */)
{
  log_message(LOG_DEBUG, "Launching Monte Carlo CPU kernel with %lu elements and %d threads.", batchSize, num_threads);
  appkernel(devPtr, batchSize, num_threads);
}

