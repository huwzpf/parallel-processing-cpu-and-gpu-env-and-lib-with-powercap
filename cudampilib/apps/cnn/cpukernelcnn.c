/*
CPU kernel for CNN staging phase.
Applies a configurable KxK median filter to each input
image (per sample), writing the result to the staging output
buffer. This simulates an image preprocessing step before GPU
inference.

Configuration:
  Compile-time define (cnn_defines.h): CPU_FILTER_SIZE (odd integer >= 1)

devPtr points to an array of two pointers (on the slave):
  [0] input  (float*)  size: batchSize * INPUT_BATCH_SIZE
  [1] output (float*)  size: batchSize * INPUT_BATCH_SIZE
*/
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define ENABLE_LOGGING
#include "logger.h"
#include "cnn_defines.h"

static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline void insertion_sort(float *a, int n) {
  for (int i = 1; i < n; ++i) {
    float key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      --j;
    }
    a[j + 1] = key;
  }
}

static void appkernel(void *devPtr, unsigned long num_elements, int num_threads)
{
  float *devPtra = (float *)(((void **)devPtr)[0]);
  float *devPtrb = (float *)(((void **)devPtr)[1]);

  const int C = CNN_IN_CHANNELS;
  const int H = CNN_IMG_H;
  const int W = CNN_IMG_W;
  const int img_stride = C * H * W; // elements per sample

  const int ksize = CPU_FILTER_SIZE;
  const int radius = CPU_FILTER_SIZE / 2;
  const int window_elems = CPU_FILTER_SIZE * CPU_FILTER_SIZE;

  // Parallelize over samples in the batch
  #pragma omp parallel for num_threads(num_threads)
  for (unsigned long b = 0; b < num_elements; ++b) {
    float *in  = devPtra + b * img_stride;
    float *out = devPtrb + b * img_stride;

    float win[CPU_FILTER_SIZE * CPU_FILTER_SIZE];

    for (int c = 0; c < C; ++c) {
      const int c_off = c * H * W;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          int idx = 0;
          // Collect KxK neighborhood with clamped borders
          for (int dy = -radius; dy <= radius; ++dy) {
            const int yy = clampi(y + dy, 0, H - 1);
            for (int dx = -radius; dx <= radius; ++dx) {
              const int xx = clampi(x + dx, 0, W - 1);
              win[idx++] = in[c_off + yy * W + xx];
            }
          }
          // Sort and take median
          insertion_sort(win, window_elems);
          out[c_off + y * W + x] = win[window_elems / 2];
        }
      }
    }
  }
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long /* id */)
{
  log_message(LOG_DEBUG, "CNN CPU: launching kernel with %lu samples and %d threads.", batchSize, num_threads);
  appkernel(devPtr, batchSize, num_threads);
}
