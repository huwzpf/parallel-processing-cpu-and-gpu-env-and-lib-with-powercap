/*
CPU kernel for CNN staging phase.
Applies a simple 3x3 box blur to each input image (per sample),
writing the result to the staging output buffer. This simulates
an image preprocessing step before GPU inference.

devPtr points to an array of two pointers (on the slave):
  [0] input  (float*)  size: batchSize * INPUT_BATCH_SIZE
  [1] output (float*)  size: batchSize * INPUT_BATCH_SIZE
*/
#include <omp.h>
#include <stdio.h>
#include <string.h>
#define ENABLE_LOGGING
#include "logger.h"
#include "cnn_defines.h"

static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static void appkernel(void *devPtr, unsigned long num_elements, int num_threads)
{
  float *devPtra = (float *)(((void **)devPtr)[0]);
  float *devPtrb = (float *)(((void **)devPtr)[1]);

  const int C = CNN_IN_CHANNELS;
  const int H = CNN_IMG_H;
  const int W = CNN_IMG_W;
  const int img_stride = C * H * W; // elements per sample

  // Parallelize over samples in the batch
  #pragma omp parallel for num_threads(num_threads)
  for (unsigned long b = 0; b < num_elements; ++b) {
    float *in  = devPtra + b * img_stride;
    float *out = devPtrb + b * img_stride;

    for (int c = 0; c < C; ++c) {
      const int c_off = c * H * W;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          float acc = 0.0f;
          int   cnt = 0;
          // 3x3 neighborhood with clamped borders
          for (int dy = -1; dy <= 1; ++dy) {
            const int yy = clampi(y + dy, 0, H - 1);
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = clampi(x + dx, 0, W - 1);
              acc += in[c_off + yy * W + xx];
              ++cnt;
            }
          }
          out[c_off + y * W + x] = acc / (float)cnt;
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
