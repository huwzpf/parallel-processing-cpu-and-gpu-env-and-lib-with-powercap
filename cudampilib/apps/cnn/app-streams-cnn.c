/*
Simple CNN app similar to RNN app.
Runs multi-layer CNN forward pass on GPU; CPU path is disabled (no-op).
*/

#include "cudampilib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "cnn_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

// Repeat data ITERS times to simulate larger memory usage without allocating it all
#define ITERS 300
#define SYNC_PERIOD 6

struct __cudampi__arguments_type __cudampi__arguments;

long long VECTORSIZE;

float *vectora; // inputs [VECTORSIZE, INPUT_BATCH_SIZE]
float *vectorb; // inputs [VECTORSIZE, INPUT_BATCH_SIZE]
float *vectorc; // outputs [VECTORSIZE, OUTPUT_BATCH_SIZE]
float *vectork; // weights  [WEIGHTS_SIZE]

unsigned long batchsize;

long long globalcounter1 = 0;
long long globalcounter2 = 0;
long long batchCounter = 0;
omp_lock_t batchCounterLock;
int streamcount = 1;
int consumeDataPoints(long long count) {
  int ret = 0;
  omp_set_lock(&batchCounterLock);
  if (batchCounter >= count) {
    batchCounter -= count;
    ret = 1;
  }
  omp_unset_lock(&batchCounterLock);

  return ret;
}

void waitForDataPoints(long long count) {
  while(!consumeDataPoints(count)) {
    usleep(100);
  }
}

void produceDataPoints(long long count) {
  omp_set_lock(&batchCounterLock);
  batchCounter += count;
  omp_unset_lock(&batchCounterLock);
}


int main(int argc, char **argv)
{
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);
  extern powercapStrategy_t __cudampi__powercapStrategy;

  if (__cudampi__powercapStrategy == BINARY_GREEDY) {
    __cudampi__powercapStrategy = EQUAL_SHARE_BINARY_GREEDY;
  }
  if (__cudampi__powercapStrategy == CONTINOUS_EQUAL) {
    __cudampi__powercapStrategy = EQUAL_SHARE_CONTINOUS_EQUAL;
  }

  streamcount = __cudampi__arguments.number_of_streams;
  batchsize = __cudampi__arguments.batch_size;
  VECTORSIZE = CNN_VECTORSIZE;

  int alldevicescount = 0;
  __cudampi__getDeviceCount(&alldevicescount);

  // Host allocations (pinned for faster transfer)
  cudaHostAlloc((void **)&vectora, sizeof(float) * VECTORSIZE * INPUT_BATCH_SIZE, cudaHostAllocDefault);
  if (!vectora)
  {
    log_message(LOG_ERROR, "Not enough memory for input.");
    exit(-1);
  }
  cudaHostAlloc((void **)&vectorb, sizeof(float) * VECTORSIZE * INPUT_BATCH_SIZE, cudaHostAllocDefault);
  if (!vectorb)
  {
    log_message(LOG_ERROR, "Not enough memory for input.");
    exit(-1);
  }
  cudaHostAlloc((void **)&vectorc, sizeof(float) * VECTORSIZE * OUTPUT_BATCH_SIZE, cudaHostAllocDefault);
  if (!vectorc)
  {
    log_message(LOG_ERROR, "Not enough memory for output.");
    exit(-1);
  }
  cudaHostAlloc((void **)&vectork, sizeof(float) * WEIGHTS_SIZE, cudaHostAllocDefault);
  if (!vectork)
  {
    log_message(LOG_ERROR, "Not enough memory for weights.");
    exit(-1);
  }

  // Init weights (shared across all layers)
  for (long long i = 0; i < WEIGHTS_SIZE; ++i)
    vectork[i] = ((float)rand()) / ((float)RAND_MAX);

  // Init inputs
  for (long long i = 0; i < (VECTORSIZE * INPUT_BATCH_SIZE); ++i)
    vectora[i] = (float)(i % 1024) * 0.001f;

  gettimeofday(&start, NULL);

  #pragma omp parallel num_threads(alldevicescount)
  {
    __cudampi__batch_pointer batch_pointer;
    int finish = 0;
    void *devPtr = NULL; // device pointer to pointer array
    void *devPtr2 = NULL; // second pointer array for stream2
    cudaStream_t stream;
    cudaStream_t stream2;
    long long privatecounter = 0;
    long long producerCounter = 0;

    int mythreadid = omp_get_thread_num();
    __cudampi__setDevice(mythreadid);
    #pragma omp barrier

    if (__cudampi__isCpu()) {
      void *devPtra = NULL, *devPtrb = NULL;
      void *devPtra2 = NULL, *devPtrb2 = NULL;
      
      __cudampi__streamCreate(&stream);
      __cudampi__malloc(&devPtra, INPUT_BATCH_SIZE * batchsize * sizeof(float));
      if (!devPtra) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtra).", mythreadid); exit(-1);}  
      __cudampi__malloc(&devPtrb, INPUT_BATCH_SIZE * batchsize * sizeof(float));
      if (!devPtrb) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrb).", mythreadid); exit(-1);} 
      __cudampi__malloc(&devPtr, 2 * sizeof(void *));
      if (!devPtr) {log_message(LOG_ERROR, "\nNot enough memory (devptr)."); exit(-1);}

      if (streamcount == 2) {
        __cudampi__streamCreate(&stream2);
        __cudampi__malloc(&devPtra2, INPUT_BATCH_SIZE * batchsize * sizeof(float));
        if (!devPtra2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtra2).", mythreadid); exit(-1);}  
        __cudampi__malloc(&devPtrb2, INPUT_BATCH_SIZE * batchsize * sizeof(float));
        if (!devPtrb2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrb2).", mythreadid); exit(-1);}   
        __cudampi__malloc(&devPtr2, 2 * sizeof(void *));
        if (!devPtr2) {log_message(LOG_ERROR, "\nNot enough memory (devptr2)."); exit(-1);}
      } 
      // Send device argument pointers
      __cudampi__memcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync(devPtr + sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream);
      if (streamcount == 2) {
        __cudampi__memcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
      }

      do {
        batch_pointer = __cudampi__getnextchunkindex(&globalcounter1, ITERS * VECTORSIZE);
        if (batch_pointer.start >= ITERS * VECTORSIZE) {
          finish = 1;
        } else {
          batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
          producerCounter += batch_pointer.n_elements;
          __cudampi__memcpyAsync(devPtra, vectora + (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);
          __cudampi__kernelInStream(devPtr, stream, 0);
          __cudampi__memcpyAsync(vectorb + (batch_pointer.start * INPUT_BATCH_SIZE), devPtrb, batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);

          if (streamcount == 2) {
            batch_pointer = __cudampi__getnextchunkindex(&globalcounter1, ITERS * VECTORSIZE);
            if (batch_pointer.start >= ITERS * VECTORSIZE) {
              finish = 1;
            } else {
              batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
              producerCounter += batch_pointer.n_elements;
              __cudampi__memcpyAsync(devPtra2, vectora + (batch_pointer.start * INPUT_BATCH_SIZE), batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream2);
              __cudampi__kernelInStream(devPtr2, stream2, 0);
              __cudampi__memcpyAsync(vectorb + (batch_pointer.start * INPUT_BATCH_SIZE), devPtrb2, batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream2);
            }
          }
        }

        privatecounter++;
        if (privatecounter % SYNC_PERIOD == 0) {
          __cudampi__deviceSynchronize();
          produceDataPoints(producerCounter);
          producerCounter = 0;
        }
      } while (!finish);

      __cudampi__deviceSynchronize();
      produceDataPoints(producerCounter);
      __cudampi__streamDestroy(stream);
      __cudampi__free(devPtr);
      __cudampi__free(devPtra);
      __cudampi__free(devPtrb);
      if (streamcount == 2) {
        __cudampi__streamDestroy(stream2);
        __cudampi__free(devPtr2);
        __cudampi__free(devPtra2);
        __cudampi__free(devPtrb2);
      }
    } else {
      // GPU thread path
      void *devPtra = NULL, *devPtrc = NULL;
      void *devPtrk = NULL, *devPtrb = NULL;
      void *devPtrw = NULL; // workspace buffer
      void *devPtra2 = NULL, *devPtrc2 = NULL;
      void *devPtrk2 = NULL, *devPtrb2 = NULL;
      void *devPtrw2 = NULL; // workspace buffer for stream2
      __cudampi__malloc(&devPtra, INPUT_BATCH_SIZE * batchsize * sizeof(float));
      if (!devPtra) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtra).", mythreadid); exit(-1);}    
      __cudampi__malloc(&devPtrc, OUTPUT_BATCH_SIZE * batchsize * sizeof(float));
      if (!devPtrc) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrc).", mythreadid); exit(-1);}   
      __cudampi__malloc(&devPtrk, WEIGHTS_SIZE * sizeof(float));
      if (!devPtrk) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrk).", mythreadid); exit(-1);}   
      __cudampi__malloc(&devPtrb, 2 * FEATURE_SIZE * batchsize * sizeof(float));
      if (!devPtrb) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrb).", mythreadid); exit(-1);}   
      // Allocate per-stream workspace once
      size_t ws_bytes = (size_t)CNN_WORKSPACE_MB * 1024ULL * 1024ULL;
      if (ws_bytes > 0) {
        __cudampi__malloc(&devPtrw, ws_bytes);
        if (!devPtrw) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrw).", mythreadid); exit(-1);} 
      }

      __cudampi__malloc(&devPtr, 5 * sizeof(void *));
      if (!devPtr) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtr).", mythreadid); exit(-1);}      

      __cudampi__streamCreate(&stream);

      // Send device argument pointers
      __cudampi__memcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync(devPtr + sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync(devPtr + 2 * sizeof(void *), &devPtrk, sizeof(void *), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync(devPtr + 3 * sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync(devPtr + 4 * sizeof(void *), &devPtrw, sizeof(void *), cudaMemcpyHostToDevice, stream);
      __cudampi__memcpyAsync(devPtrk, vectork, WEIGHTS_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

      if (streamcount == 2) {
        // Allocate second set for the second stream
        __cudampi__malloc(&devPtra2, INPUT_BATCH_SIZE * batchsize * sizeof(float));
        if (!devPtra2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtra2).", mythreadid); exit(-1);}    
        __cudampi__malloc(&devPtrc2, OUTPUT_BATCH_SIZE * batchsize * sizeof(float));
        if (!devPtrc2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrc2).", mythreadid); exit(-1);}   
        __cudampi__malloc(&devPtrk2, WEIGHTS_SIZE * sizeof(float));
        if (!devPtrk2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrk2).", mythreadid); exit(-1);}   
        __cudampi__malloc(&devPtrb2, 2 * FEATURE_SIZE * batchsize * sizeof(float));
        if (!devPtrb2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrb2).", mythreadid); exit(-1);}   
        // workspace for stream2
        if (ws_bytes > 0) {
          __cudampi__malloc(&devPtrw2, ws_bytes);
          if (!devPtrw2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtrw2).", mythreadid); exit(-1);} 
        }
        __cudampi__malloc(&devPtr2, 5 * sizeof(void *));
        if (!devPtr2) { log_message(LOG_ERROR, "[T%d] Not enough memory (devPtr2).", mythreadid); exit(-1);}      

        __cudampi__streamCreate(&stream2);
        __cudampi__memcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + 2 * sizeof(void *), &devPtrk2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + 3 * sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + 4 * sizeof(void *), &devPtrw2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtrk2, vectork, WEIGHTS_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream2);
      }

      do {
        batch_pointer = __cudampi__getnextchunkindex(&globalcounter2, ITERS * VECTORSIZE);
        if (batch_pointer.start >= ITERS * VECTORSIZE) {
          finish = 1;
        } else {  
          waitForDataPoints(batch_pointer.n_elements);
          // Map the virtual index space to the real buffer range
          batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
          // Copy inputs for this chunk
          __cudampi__memcpyAsync(
            devPtra,
            vectorb + (batch_pointer.start * INPUT_BATCH_SIZE),
            batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(float),
            cudaMemcpyHostToDevice,
            stream);

          // Run CNN forward
          __cudampi__kernelInStream(devPtr, stream, 0);

          // Copy outputs back
          __cudampi__memcpyAsync(
            vectorc + (batch_pointer.start * OUTPUT_BATCH_SIZE),
            devPtrc,
            batch_pointer.n_elements * OUTPUT_BATCH_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream);

          if (streamcount == 2) {
            // Schedule second chunk to stream2
            batch_pointer = __cudampi__getnextchunkindex(&globalcounter2, ITERS * VECTORSIZE);
            if (batch_pointer.start >= ITERS * VECTORSIZE) {
              finish = 1;
            } else {
              waitForDataPoints(batch_pointer.n_elements);
              batch_pointer.start = batch_pointer.start % (VECTORSIZE - batchsize);
              __cudampi__memcpyAsync(
                devPtra2,
                vectorb + (batch_pointer.start * INPUT_BATCH_SIZE),
                batch_pointer.n_elements * INPUT_BATCH_SIZE * sizeof(float),
                cudaMemcpyHostToDevice,
                stream2);
              __cudampi__kernelInStream(devPtr2, stream2, 0);
              __cudampi__memcpyAsync(
                vectorc + (batch_pointer.start * OUTPUT_BATCH_SIZE),
                devPtrc2,
                batch_pointer.n_elements * OUTPUT_BATCH_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream2);
            }
          }
        }

        privatecounter++;
        if (privatecounter % SYNC_PERIOD == 0) {
          __cudampi__deviceSynchronize();
        }
      } while (!finish);
      __cudampi__deviceSynchronize();
      __cudampi__streamDestroy(stream);
      __cudampi__free(devPtr);
      __cudampi__free(devPtra);
      __cudampi__free(devPtrc);
      __cudampi__free(devPtrk);
      __cudampi__free(devPtrb);
      if (devPtrw) __cudampi__free(devPtrw);
      if (streamcount == 2) {
        __cudampi__streamDestroy(stream2);
        __cudampi__free(devPtr2);
        __cudampi__free(devPtra2);
        __cudampi__free(devPtrc2);
        __cudampi__free(devPtrk2);
        __cudampi__free(devPtrb2);
        if (devPtrw2) __cudampi__free(devPtrw2);
      }
    }
  }

  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  __cudampi__terminateMPI();

  // Optionally persist outputs for quick verification
  // save_vector_output_double(vectorc, VECTORSIZE, "CNN_logs_cpugpuasyncfull.log", "CPUGPUASYNC");

  cudaFreeHost(vectora);
  cudaFreeHost(vectorb);
  cudaFreeHost(vectorc);
  cudaFreeHost(vectork);

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
