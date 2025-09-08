/*
CPU side: basic preprocessing to simulate host work.
Given pointers [input_raw, output], perform per-pixel normalization and optional flip.
This kernel is different from the GPU kernel on purpose (preproc vs conv).
*/
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "vision_defines.h"

static inline float clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void preprocess_batch(const float* in_raw, float* out_prep, unsigned long batchSize) {
  const int C = VISION_IMG_C, H = VISION_IMG_H, W = VISION_IMG_W;
  const size_t per = (size_t)C * H * W;
  // Simple per-pixel ops: scale to [0,1], normalize, optional horizontal flip based on sample index
  #pragma omp parallel for schedule(static)
  for (unsigned long n = 0; n < batchSize; ++n) {
    const float* src = in_raw + n * per;
    float* dst = out_prep + n * per;
    int flip = ((n & 1u) == 0u); // flip every other sample deterministically
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          int sx = flip ? (W - 1 - x) : x;
          size_t si = ((size_t)c * H + y) * W + sx;
          size_t di = ((size_t)c * H + y) * W + x;
          float v = src[si];
          v = clamp(v, 0.f, 255.f) * (1.f / 255.f); // to [0,1]
          // channel-wise fake mean/std (for demo)
          float mean = 0.5f + 0.1f * c;
          float std = 0.25f + 0.05f * c;
          dst[di] = (v - mean) / std;
        }
      }
    }
  }
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long /* id */) {
  omp_set_num_threads(num_threads);
  log_message(LOG_DEBUG, "Launching CPU preproc with %lu samples and %d threads.", batchSize, num_threads);

  float *in_raw = (float *)(((void **)devPtr)[0]);
  float *out    = (float *)(((void **)devPtr)[1]);
  // Reuse output buffer to store preprocessed data in CPU-only path
  preprocess_batch(in_raw, out, batchSize);
}

