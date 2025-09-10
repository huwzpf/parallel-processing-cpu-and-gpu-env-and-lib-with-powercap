/*
Empty CPU kernel stub for CNN app. No computations are performed on CPU.
*/
#include <stdio.h>
#define ENABLE_LOGGING
#include "logger.h"

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long /* id */)
{
  (void)devPtr; (void)batchSize; (void)num_threads;
  // Intentionally no-op
  log_message(LOG_DEBUG, "CNN CPU kernel called as no-op (batch=%lu, threads=%d)", batchSize, num_threads);
}

