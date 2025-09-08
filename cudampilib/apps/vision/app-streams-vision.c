/*
Vision app: CPU performs preprocessing (decode/normalize/flip-like), GPU runs a conv layer.
Modeled on other app-streams-*.c files with double-buffered streams option.
*/
#include "cudampilib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "vision_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

struct __cudampi__arguments_type __cudampi__arguments;

long long VECTORSIZE; // number of samples (images)

// Host buffers
float *images_raw;     // VECTORSIZE x INPUT_SIZE
float *images_prep;    // VECTORSIZE x INPUT_SIZE (preprocessed)
float *weights;        // WEIGHTS_SIZE
float *outputs;        // VECTORSIZE x OUTPUT_SIZE

unsigned long batchsize;
// Shared counters for batch distribution/checks
long long globalcounter = 0;          // CPU preprocessing global counter
long long gpu_dummy_counter = 0;      // GPU-side dummy counter (enable check only)
int streamcount = 1;

static inline float frand() { return (float)rand() / (float)RAND_MAX; }

// Ready-queue from CPU preprocessors -> GPU consumers
typedef struct { long long start; unsigned long n; } ReadyItem;
#define READY_Q_CAP 1024
static ReadyItem ready_q[READY_Q_CAP];
static int ready_head = 0, ready_tail = 0, ready_count = 0;
static omp_lock_t ready_lock;
static int producers_active = 0;      // number of active CPU producer threads

static int ready_push(long long start, unsigned long n) {
  int ok = 0;
  omp_set_lock(&ready_lock);
  if (ready_count < READY_Q_CAP) {
    ready_q[ready_tail] = (ReadyItem){start, n};
    ready_tail = (ready_tail + 1) % READY_Q_CAP;
    ready_count++;
    ok = 1;
  }
  omp_unset_lock(&ready_lock);
  return ok;
}

static int ready_pop(ReadyItem* out) {
  int ok = 0;
  omp_set_lock(&ready_lock);
  if (ready_count > 0) {
    *out = ready_q[ready_head];
    ready_head = (ready_head + 1) % READY_Q_CAP;
    ready_count--;
    ok = 1;
  }
  omp_unset_lock(&ready_lock);
  return ok;
}

int main(int argc, char **argv) {
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;
  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  streamcount = __cudampi__arguments.number_of_streams;
  batchsize   = __cudampi__arguments.batch_size;
  VECTORSIZE  = VISION_NUM_SAMPLES;

  assert(batchsize > 0);

  int alldevicescount = 0;
  __cudampi__getDeviceCount(&alldevicescount);

  // Allocate host pinned buffers
  size_t in_total = (size_t)VECTORSIZE * VISION_INPUT_SIZE;
  size_t out_total = (size_t)VECTORSIZE * VISION_OUTPUT_SIZE;
  cudaHostAlloc((void **)&images_raw,  sizeof(float) * in_total, cudaHostAllocDefault);
  if (!images_raw) { log_message(LOG_ERROR, "Not enough memory for images_raw."); exit(-1); }
  cudaHostAlloc((void **)&images_prep, sizeof(float) * in_total, cudaHostAllocDefault);
  if (!images_prep) { log_message(LOG_ERROR, "Not enough memory for images_prep."); exit(-1); }
  cudaHostAlloc((void **)&outputs,     sizeof(float) * out_total, cudaHostAllocDefault);
  if (!outputs)     { log_message(LOG_ERROR, "Not enough memory for outputs."); exit(-1); }
  cudaHostAlloc((void **)&weights,     sizeof(float) * VISION_WEIGHTS_SIZE, cudaHostAllocDefault);
  if (!weights)     { log_message(LOG_ERROR, "Not enough memory for weights."); exit(-1); }

  // Initialize raw images with synthetic data and conv weights
  for (size_t i = 0; i < in_total; ++i) {
    images_raw[i] = frand() * 255.0f;
  }
  for (int i = 0; i < VISION_WEIGHTS_SIZE; ++i) {
    weights[i] = (frand() - 0.5f) * 0.1f; // small kernel values
  }

  // Initialize queue lock
  omp_init_lock(&ready_lock);

  gettimeofday(&start, NULL);

  #pragma omp parallel num_threads(alldevicescount)
  {
    int mythreadid = omp_get_thread_num();
    __cudampi__setDevice(mythreadid);
    #pragma omp barrier

    if (__cudampi__isCpu()) {
      // CPU producer: preprocess host -> host(images_prep) via remote CPU kernel
      void *devIn = NULL, *devOut = NULL, *devPtr = NULL;
      cudaStream_t stream;

      #pragma omp atomic
      producers_active++;

      __cudampi__malloc(&devIn,  batchsize * VISION_INPUT_SIZE * sizeof(float));
      __cudampi__malloc(&devOut, batchsize * VISION_INPUT_SIZE * sizeof(float));
      __cudampi__malloc(&devPtr, 2 * sizeof(void*));
      __cudampi__streamCreate(&stream);
      __cudampi__memcpyAsync(devPtr, &devIn,  sizeof(void*), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync((char*)devPtr + sizeof(void*), &devOut, sizeof(void*), cudaMemcpyHostToDevice, stream);

      unsigned long produced_items = 0, produced_samples = 0;
      while (1) {
        // Fetch next preproc range using cudampi counter (CPU batch sizing)
        __cudampi__batch_pointer batch_pointer = __cudampi__getnextchunkindex(&globalcounter, VECTORSIZE);
        if (batch_pointer.start >= VECTORSIZE) {
          break;
        }
        long long start = batch_pointer.start;
        unsigned long n = batch_pointer.n_elements;

        // H2D raw -> CPU device input
        __cudampi__memcpyAsync(devIn,
          images_raw + (size_t)start * VISION_INPUT_SIZE,
          n * VISION_INPUT_SIZE * sizeof(float),
          cudaMemcpyHostToDevice, stream);
        // Launch CPU preprocess kernel
        __cudampi__kernelInStream(devPtr, stream, 0);
        // D2H CPU device output -> images_prep
        __cudampi__memcpyAsync(
          images_prep + (size_t)start * VISION_INPUT_SIZE,
          devOut,
          n * VISION_INPUT_SIZE * sizeof(float),
          cudaMemcpyDeviceToHost, stream);
        __cudampi__deviceSynchronize();

        // Push ready range for GPU
        // Busy-wait if queue full
        while (!ready_push(start, n)) { /* queue full */ }
        produced_items += 1; produced_samples += n;
      }

      __cudampi__streamDestroy(stream);
      __cudampi__free(devIn); __cudampi__free(devOut); __cudampi__free(devPtr);

      #pragma omp atomic
      producers_active--;
    } else {
      // GPU consumer: conv on preprocessed data from images_prep
      void *devIn = NULL, *devW = NULL, *devOut = NULL, *devPtr = NULL;
      cudaStream_t stream;

      __cudampi__malloc(&devIn,  batchsize * VISION_INPUT_SIZE  * sizeof(float));
      __cudampi__malloc(&devW,   VISION_WEIGHTS_SIZE            * sizeof(float));
      __cudampi__malloc(&devOut, batchsize * VISION_OUTPUT_SIZE * sizeof(float));
      __cudampi__malloc(&devPtr, 3 * sizeof(void*));
      __cudampi__streamCreate(&stream);
      __cudampi__memcpyAsync(devPtr, &devIn,  sizeof(void*), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync((char*)devPtr + sizeof(void*), &devW,  sizeof(void*), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync((char*)devPtr + 2*sizeof(void*), &devOut, sizeof(void*), cudaMemcpyHostToDevice, stream);
      // Upload weights once
      __cudampi__memcpyAsync(devW, weights, VISION_WEIGHTS_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

      long long consumed = 0;
      unsigned long consumed_items = 0, consumed_samples = 0;
      while (1) {
        ReadyItem item;
        if (!ready_pop(&item)) {
          // Check termination: no producers and queue empty
          int prod;
          #pragma omp atomic read
          prod = producers_active;
          if (prod == 0) {
            break; // done
          }
          continue; // spin
        }
        
        __cudampi__batch_pointer chk = __cudampi__getnextchunkindex(&gpu_dummy_counter, VECTORSIZE);

        if (chk.start >= VECTORSIZE) {
          break;
        }

        consumed += item.n;
        consumed_items += 1; consumed_samples += item.n;
        // Process this range
        __cudampi__memcpyAsync(devIn,
          images_prep + (size_t)item.start * VISION_INPUT_SIZE,
          item.n * VISION_INPUT_SIZE * sizeof(float),
          cudaMemcpyHostToDevice, stream);
        __cudampi__kernelInStream(devPtr, stream, 0);
        __cudampi__memcpyAsync(
          outputs + (size_t)item.start * VISION_OUTPUT_SIZE,
          devOut,
          item.n * VISION_OUTPUT_SIZE * sizeof(float),
          cudaMemcpyDeviceToHost, stream);
        __cudampi__deviceSynchronize();
      }

      __cudampi__streamDestroy(stream);
      __cudampi__free(devIn); __cudampi__free(devW); __cudampi__free(devOut); __cudampi__free(devPtr);
    }
  }

  gettimeofday(&stop, NULL);

  double t = (stop.tv_sec - start.tv_sec) + 1e-6 * (stop.tv_usec - start.tv_usec);
  log_message(LOG_INFO, "Main elapsed time=%lf", t);

  __cudampi__terminateMPI();
  gettimeofday(&stoptotal, NULL);
  double ttotal = (stoptotal.tv_sec - starttotal.tv_sec) + 1e-6 * (stoptotal.tv_usec - starttotal.tv_usec);
  log_message(LOG_INFO, "Total elapsed time=%lf", ttotal);

  return 0;
}
