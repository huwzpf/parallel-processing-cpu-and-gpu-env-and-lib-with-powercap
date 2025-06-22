/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <argp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <nvml.h>
#include <omp.h>
#include <assert.h>

#define ENABLE_LOGGING
#define MPI_LOGGING
#include "logger.h"

#include "cudampicommon.h"
#include "cudampilib.h"

// int __cudampi__GPUcountpernode=1;

#define __cudampi__currentDevice  __cudampi__currentdevice[omp_get_thread_num()]
#define __cudampi__currentCommunicator  __cudampi__communicators[__cudampi__currentDevice]
#define  __cudampi_isLocalGpu __cudampi__currentDevice < __cudampi__GPUcountspernode[0]
#define __cudampi__currentMemcpyQueue &(__cudampi__memcpy_queues[omp_get_thread_num()])

int *__cudampi__GPUcountspernode;
int *__cudampi__CPUcountspernode;
int *__cudampi__freeThreadsPerNode;
int __cudampi_totaldevicecount = 0; // how many GPUs + CPUs (on all considered nodes)
int __cudampi_totalgpudevicecount = 0; // how many GPUs in total (on all considered nodes)
int __cudampi_totalcpudevicecount = 0; // how many CPUs in total (on all considered nodes)

int __cudampi__localGpuDeviceCount = 0; // how many GPUs in process 0
int __cudampi__localFreeThreadCount = 0;

int *__cudampi_targetGPUfordevice;     // GPU id on a given node for device number (global)
int *__cudampi_targetMPIrankfordevice; // MPI rank for device number (global)

int __cudampi__MPIinitialized = 0;
int __cudampi__MPIproccount;
int __cudampi__myrank;

double __cudampi__totalEnergyUsed = 0;
int __cudampi__dyanmicCpuBatchSizeScalingEnabled;

double __cudampi__firstIterTotalGpuBatchTimeSeconds = 0;
double __cudampi__firstIterTotalCpuBatchTimeSeconds = 0;
unsigned long __cudampi__firstIterTotalGpuBatches = 0;
unsigned long __cudampi__firstIterTotalCpuBatches = 0;
int __cudampi__firstIterNumberGpuDevicesMeasured = 0;
int __cudampi__firstIterNumberCpuDevicesMeasured = 0;
int __cudampi__firstIterDeviceMeasurementStarted[__CUDAMPI_MAX_THREAD_COUNT] = {0};
int __cudampi__firstIterMeasuredForDevice[__CUDAMPI_MAX_THREAD_COUNT] = {0};
int __cudampi__cpuBatchSizeScalingDone = 0;

float cpuLastEnergyMeasured[MAX_THREADS] = {0.0};
omp_lock_t cpuEnergyLock[MAX_THREADS];
int isInitialCpuEnergyMeasured[MAX_THREADS] = {0};

int __cudampi__currentdevice[__CUDAMPI_MAX_THREAD_COUNT]; // current device id for various threads in process 0

struct timeval __cudampi__timestart[__CUDAMPI_MAX_THREAD_COUNT]; // start of time measurement
struct timeval __cudampi__timestop[__CUDAMPI_MAX_THREAD_COUNT];  // end of time measurement
double __cudampi__time_us[__CUDAMPI_MAX_THREAD_COUNT];      // last time measurement (from start to stop)
int __cudampi__timemeasured[__CUDAMPI_MAX_THREAD_COUNT] = {0};   // whether time measurement started
float __cudampi__devicepower[__CUDAMPI_MAX_THREAD_COUNT];        // current power taken by a device

int __cudampi__deviceenabled[__CUDAMPI_MAX_THREAD_COUNT]; // whether the given device is enabled for further use

omp_lock_t __cudampi__devicelocks[__CUDAMPI_MAX_THREAD_COUNT]; // locks that guard writing to and reading from power and time values for particular devices

omp_lock_t deviceselectionlock;

int __cudampi__amimanager[__CUDAMPI_MAX_THREAD_COUNT] = {0}; // whether the given thread/device is the manager for device selection

MPI_Comm *__cudampi__communicators; // communicators for communication with threads responsible for target GPUs, there is one communicator for such target GPU

int __cudampi__isglobalpowerlimitset = 0; // whether a global power limit has been set
float __cudampi__globalpowerlimit;

int powermeasurecounter[__CUDAMPI_MAX_THREAD_COUNT] = {0};

unsigned long __cudampi__batches_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long long __cudampi__data_points_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long __cudampi__last_batches_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long long __cudampi__last_data_points_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};

unsigned long __cudampi__default_batch_size;
unsigned long __cudampi__cpu_batch_size;
float __cudampi__cpu_power_scaling;

int __cudampi__cpu_enabled;
extern struct __cudampi__arguments_type __cudampi__arguments;

static char doc[] = "Cudampi program";
static char args_doc[] = "";
static struct argp_option options[] = {
  { "cpu-enabled",                       'c',  "ENABLED",           0, "Enable CPU processing (1 to enable, 0 to disable)" },
  { "number-of-streams",                 'n',  "NUM",               0, "Set the number of streams" },
  { "batch-size",                        'b',  "SIZE",              0, "Set the batch size" },
  { "powercap",                          'p',  "WATTS",             0, "Set the power cap (0 to disable)" },
  { "cpu-power-scaling",                 's',  "SCALING FACTOR",    0, "Set the CPU power scaling factor" },
  { "initial-cpu-batch-size-scaling",    'f',  "SCALING FACTOR",    0, "Set initial scaling factor for CPU batch size (0 to disable)" },
  { "disable-dynamic-cpu-batch-scaling",  0,    0,                  0, "Disable dynamic CPU batch size scaling" },
  { 0 }
};

// Forward declaration for argp definition
static error_t parse_opt(int key, char *arg, struct argp_state *state);

static struct argp argp = { options, parse_opt, args_doc, doc };

// Counter that holds a unique tag for asynchronously exchanged messages
// it increments by 2 (D_MSG_TAG) to accomodate data message and status
int asyncMsgCounter = MIN_ASYNC_MSG_TAG;

typedef struct memcpy_queue_entry {
    MPI_Request dataRequest;
    MPI_Request statusRequest;
    cudaError_t status;
    TAILQ_ENTRY(memcpy_queue_entry) entries;
} memcpy_queue_entry_t;

// Define the queue head
TAILQ_HEAD(memcpy_queue_head, memcpy_queue_entry);

// Declare queues for memcpy operations
struct memcpy_queue_head __cudampi__memcpy_queues[__CUDAMPI_MAX_THREAD_COUNT];

unsigned long __cudampi__getCurrentBatchSize() {
  unsigned long ret = 0;

  if (__cudampi__isCpu()) {
    #pragma omp atomic read
    ret = __cudampi__cpu_batch_size;
  }
  else {
    ret = __cudampi__default_batch_size;
  }
  return ret;
}

// Start measurement of first iteration (between first kernel call and first synchronize call) performance for each device
// This function starts measurement when first kernel is called and increments amount of processed data with each kernel call
void __cudampi__recordBatchSizeForDeviceStats(unsigned long batchsize) {
  if (!__cudampi__dyanmicCpuBatchSizeScalingEnabled) {
    return;
  }
  // If this is the first iteration (deviceSynchronize was not called yet)
  if (!__cudampi__firstIterMeasuredForDevice[__cudampi__currentDevice]) {
    // If this is the first call to this function in given thread
    if (!__cudampi__firstIterDeviceMeasurementStarted[__cudampi__currentDevice]) {
      // Record time when kernel was first called for this thread
      __cudampi__firstIterDeviceMeasurementStarted[__cudampi__currentDevice] = 1;
      __cudampi__timemeasured[__cudampi__currentDevice] = 1;
      gettimeofday(&(__cudampi__timestart[__cudampi__currentDevice]), NULL);
    }
    if (__cudampi__isCpu()) {
      #pragma omp atomic
      __cudampi__firstIterTotalCpuBatches += batchsize;
    } else {
      #pragma omp atomic
      __cudampi__firstIterTotalGpuBatches += batchsize;
    }
  }
}

// This function will be called after every synchronize call, but will actually only have effect in first call
void __cudampi__finishDeviceStatsMeasurement(double elapsedTimeSeconds) {
  if (!__cudampi__dyanmicCpuBatchSizeScalingEnabled) {
    return;
  }

  if (!__cudampi__firstIterMeasuredForDevice[__cudampi__currentDevice]) {
    __cudampi__firstIterMeasuredForDevice[__cudampi__currentDevice] = 1;

    if (!__cudampi__firstIterDeviceMeasurementStarted[__cudampi__currentDevice]) {
      log_message(LOG_ERROR, "Synchronization before first kernel call. Dynamic batch size scaling won't work.");
      return;
    }

    if (__cudampi__isCpu()) {
      #pragma omp atomic
      __cudampi__firstIterNumberCpuDevicesMeasured++;

      #pragma omp atomic
      __cudampi__firstIterTotalCpuBatchTimeSeconds += elapsedTimeSeconds;
    } else {
      #pragma omp atomic
      __cudampi__firstIterNumberGpuDevicesMeasured++;

      #pragma omp atomic
      __cudampi__firstIterTotalGpuBatchTimeSeconds += elapsedTimeSeconds;
    }
  }
}

// Check if all devices already finished first iteration
int __cudampi__readyForCpuBatchSizeScaling() {
  int nCpuMeasured;
  int nGpuMeasured;

  #pragma omp atomic read
  nCpuMeasured = __cudampi__firstIterNumberCpuDevicesMeasured;

  #pragma omp atomic read
  nGpuMeasured = __cudampi__firstIterNumberGpuDevicesMeasured;

  return ((nCpuMeasured >= __cudampi_totalcpudevicecount) && (nGpuMeasured >= __cudampi_totalgpudevicecount));
}

void __cudampi__scaleCpuBatchSize() {
  if (!__cudampi__dyanmicCpuBatchSizeScalingEnabled) {
    return;
  }
  int scalingDone;

  #pragma omp atomic read
  scalingDone = __cudampi__cpuBatchSizeScalingDone;

  if (scalingDone || (!__cudampi__isCpu()) || (!__cudampi__readyForCpuBatchSizeScaling())) {
    return;
  }
  // Here we are sure that this thread is CPU and first iteration already finished but scaling was not done
  #pragma omp critical
  {
    #pragma omp atomic read
    scalingDone = __cudampi__cpuBatchSizeScalingDone;

    // Make sure that some thread did not execute scaling between previous check and critical section
    if (!scalingDone) {
      // Code below will be executed by first CPU thread that reaches this function after synchronize was called for each device at least once

      // Scale cpu batch size by:
      // scalingFactor = gpuBatchTimePerBatch / cpuBatchTimePerBatch
      double scalingFactor = (__cudampi__firstIterTotalGpuBatchTimeSeconds / __cudampi__firstIterTotalGpuBatches) / (__cudampi__firstIterTotalCpuBatchTimeSeconds / __cudampi__firstIterTotalCpuBatches);

      // Don't scale cpu batch size up as it would exceed allocated buffers
      if (scalingFactor < 1) {
        long long newCpuBatchSize = (long long)((double)(__cudampi__default_batch_size)*scalingFactor);

        // Make sure that at least one batch is delegated to CPU
        if (newCpuBatchSize < 1) {
          newCpuBatchSize = 1;
        }

        #pragma omp atomic write
        __cudampi__cpu_batch_size = (unsigned long)newCpuBatchSize;

        log_message(LOG_INFO, "Scaling CPU batch size by %lf (new value: %lld)", scalingFactor, newCpuBatchSize);
      }

      // Atomic because other threads might be reading it
      #pragma omp atomic write
      __cudampi__cpuBatchSizeScalingDone = 1;
    }
  }
}

int getMsgCounter() {
  int result;
  #pragma omp critical
  {
    if (asyncMsgCounter >= MAX_ASYNC_MSG_TAG)
    {
      asyncMsgCounter = MIN_ASYNC_MSG_TAG;
    }
    result = asyncMsgCounter;
    asyncMsgCounter += D_MSG_TAG;
  }
  return result;
}

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
  struct __cudampi__arguments_type *arguments = state->input;

  switch (key)
    {
    case 'c':
      arguments->cpu_enabled = atoi(arg);
      break;
    case 'n':
      arguments->number_of_streams = atoi(arg);
      break;
    case 'b':
      arguments->batch_size = atol(arg);
      break;
    case 'p':
      arguments->powercap = atoi(arg);
      break;
    case 's':
      arguments->cpu_power_scaling = atof(arg);
      break;
    case 'f':
      arguments->cpu_batch_scaling_factor = atoi(arg);
      break;
    case 0: // For --disable-dynamic-cpu-batch-scaling
      arguments->use_dynamic_scaling = 0;
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

void initiateAsyncRecv(void* dst, unsigned long count, int counter)
{ 
  memcpy_queue_entry_t *item = malloc(sizeof(memcpy_queue_entry_t));
  if (item == NULL) {
      log_message(LOG_ERROR, "Failed to allocate memory");
  }

  // Receive the data
  MPI_Irecv(dst, count, MPI_UNSIGNED_CHAR, 1, counter , __cudampi__currentCommunicator, &item->dataRequest);

  // Receive the status code
  MPI_Irecv((unsigned char*)(&item->status), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 1, counter + 1, __cudampi__currentCommunicator,  &item->statusRequest);

  TAILQ_INSERT_TAIL(__cudampi__currentMemcpyQueue, item, entries);
}

void initiateAsyncSendCpu(const void* src, unsigned long count, int counter)
{ 
  // For CPU, asynchronously send data to device and asynchronously wait for response
  memcpy_queue_entry_t *item = malloc(sizeof(memcpy_queue_entry_t));
  if (item == NULL) {
      log_message(LOG_ERROR, "Failed to allocate memory");
  }
  // Send the data
  MPI_Isend(src, count, MPI_UNSIGNED_CHAR, 1, counter, __cudampi__currentCommunicator, &item->dataRequest);

  // Receive the status code
  MPI_Irecv((unsigned char*)(&item->status), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 1, counter + 1, __cudampi__currentCommunicator,  &item->statusRequest);

  TAILQ_INSERT_TAIL(__cudampi__currentMemcpyQueue, item, entries);
}

void waitForAsyncSendResponse(int counter)
{ 
  // For GPU, synchronously send data to device and asynchronously wait for response
  memcpy_queue_entry_t *item = malloc(sizeof(memcpy_queue_entry_t));
  if (item == NULL) {
      log_message(LOG_ERROR, "Failed to allocate memory");
  }
  // Send the data
  item->dataRequest = NULL;

  // Receive the status code
  MPI_Irecv((unsigned char*)(&item->status), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 1, counter + 1, __cudampi__currentCommunicator,  &item->statusRequest);

  TAILQ_INSERT_TAIL(__cudampi__currentMemcpyQueue, item, entries);
}

void process_queue() {
    memcpy_queue_entry_t* item;
    MPI_Status status;
    // Process all items in the queue
    while (!TAILQ_EMPTY(__cudampi__currentMemcpyQueue)) {
        item = TAILQ_FIRST(__cudampi__currentMemcpyQueue);

        // Wait for the request data to be received / sent
        if (item->dataRequest != NULL) {
          MPI_Wait(&item->dataRequest, &status);
        }
        MPI_Wait(&item->statusRequest, &status);

        if(item->status != cudaSuccess)
        {
          log_message(LOG_ERROR, "Received non zero status code: %d", item->status);
        }

        TAILQ_REMOVE(__cudampi__currentMemcpyQueue, item, entries);
    }
}

void __cudampi__setglobalpowerlimit(float powerlimit) {

  __cudampi__isglobalpowerlimitset = 1;
  __cudampi__globalpowerlimit = powerlimit;
}


float __cudampi__gettotalpowerofselecteddevices() { // gets total power of currently enabled devices
  int i;
  float power = 0;
  float curpower;

  omp_set_lock(&deviceselectionlock);

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
  }

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__deviceenabled[__cudampi__currentdevice[i]] == 1) {
      curpower = __cudampi__devicepower[__cudampi__currentdevice[i]];
      if (curpower == (-1)) {

        for (int i = 0; i < __cudampi_totaldevicecount; i++) {
          omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
        }

        omp_unset_lock(&deviceselectionlock);
        return -1;
      }
      power += curpower;
    }
  }

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
  }

  omp_unset_lock(&deviceselectionlock);
  return power;
}

int __cudampi__selectdevicesforpowerlimit_greedy() { // adopts a greedy strategy for selecting devices
                                                     // returns 1 if successful, 0 otherwise - if not all devices have been recorder power
  int i;
  float powerleft;
  int indexselected;
  float curperfpower;
  int anydeviceenabled = 0;

  log_message(LOG_DEBUG, "\nBefore setting power cap");
  fflush(stdout);

  omp_set_lock(&deviceselectionlock);

  if (__cudampi__isglobalpowerlimitset == 0) {
    log_message(LOG_DEBUG,"\nPowercap has not been set");
    fflush(stdout);
    omp_unset_lock(&deviceselectionlock);
    return 0;
  }

  powerleft = __cudampi__globalpowerlimit;
  // this will be invoked from one thread typically
  fflush(stdout);
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    log_message(LOG_DEBUG,"\nSetting lock on %d %d", i, __cudampi__currentdevice[i]);
    fflush(stdout);
    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
    // disable all devices at first
  }
  fflush(stdout);

  // check of all the devices has been set power
  int allpowerset = 1;
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicepower[__cudampi__currentdevice[i]] == (-1)) {
      allpowerset = 0;
      break;
    }
  }

  if (!allpowerset) {
    // unlock and quit
    log_message(LOG_DEBUG,"Before setting powercap");
    fflush(stdout);
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
    }
    log_message(LOG_DEBUG,"After setting powercap");
    fflush(stdout);

    omp_unset_lock(&deviceselectionlock);

    return 0;
  }

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__deviceenabled[__cudampi__currentdevice[i]] == 1) {
      __cudampi__deviceenabled[__cudampi__currentdevice[i]] = -1; // candidate for selection
    }
    __cudampi__amimanager[__cudampi__currentdevice[i]] = 0;
  }

  fflush(stdout);
  int managerselected = 0;
  do {
    curperfpower = 0;
    indexselected = -1;
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      
      float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[i]) / __cudampi__devicepower[i];
      if (((-1) == (__cudampi__deviceenabled[__cudampi__currentdevice[i]])) && (__cudampi__devicepower[__cudampi__currentdevice[i]] <= powerleft) &&
          (inverseDeviceEnergyUsed > curperfpower)) {
        curperfpower = inverseDeviceEnergyUsed;
        indexselected = i;
        anydeviceenabled = 1;
      }
    }
    if (indexselected != (-1)) {
      // enable the found device now
      __cudampi__deviceenabled[__cudampi__currentdevice[indexselected]] = 1;
      if (!managerselected) {
        managerselected = 1;
        __cudampi__amimanager[__cudampi__currentdevice[indexselected]] = 1;
      }

      // Log if the selected device is CPU or GPU and how much power it subtracts
      if (__cudampi__currentdevice[indexselected] >= __cudampi_totalgpudevicecount) {
        if(__cudampi__devicepower[__cudampi__currentdevice[indexselected]] <= 0)
        {
          log_message(LOG_ERROR,"Selected CPU device %d is not correctly calculated", __cudampi__currentdevice[indexselected]);
        }
        log_message(LOG_INFO, "Selected CPU device %d, subtracted power: %f, inverse device energy used: %f", __cudampi__currentdevice[indexselected], __cudampi__devicepower[__cudampi__currentdevice[indexselected]], curperfpower);
      } else {
        log_message(LOG_INFO, "Selected GPU device %d, subtracted power: %f, inverse device energy used: %f", __cudampi__currentdevice[indexselected], __cudampi__devicepower[__cudampi__currentdevice[indexselected]], curperfpower);
      }

      // Log devices that were not selected and their power
      for (int j = 0; j < __cudampi_totaldevicecount; j++) {
        if (__cudampi__deviceenabled[__cudampi__currentdevice[j]] != 1) {
          float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[j]) / __cudampi__devicepower[j];
          if (__cudampi__currentdevice[j] >= __cudampi_totalgpudevicecount) {
        log_message(LOG_INFO, "Not selected CPU device %d, power: %f, inverse device energy used: %f", __cudampi__currentdevice[j], __cudampi__devicepower[__cudampi__currentdevice[j]], inverseDeviceEnergyUsed);
          } else {
        log_message(LOG_INFO, "Not selected GPU device %d, power: %f, inverse device energy used: %f", __cudampi__currentdevice[j], __cudampi__devicepower[__cudampi__currentdevice[j]], inverseDeviceEnergyUsed);
          }
        }
      }


      powerleft -= __cudampi__devicepower[__cudampi__currentdevice[indexselected]];
      log_message(LOG_DEBUG,"\nSelected device %d", __cudampi__currentdevice[indexselected]);
      log_message(LOG_INFO, "Remaining power left: %f", powerleft);
    }
  } while (indexselected != (-1));
  fflush(stdout);

  if (!anydeviceenabled) { // handle this case
    log_message(LOG_ERROR,"No devices found under the power limit");
    fflush(stdout);
    exit(-1);
  }

  // now not enabled devices are set to 0
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__deviceenabled[__cudampi__currentdevice[i]] != 1) {
      __cudampi__deviceenabled[__cudampi__currentdevice[i]] = 0;
    }
  }

   if (powerleft > 0) {
    log_message(LOG_INFO, "All devices are selected and there is still power left: %f", powerleft);
  }

  // unlock the devices' locks
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
  }

  fflush(stdout);

  omp_unset_lock(&deviceselectionlock);

  return 1;
}

__cudampi__batch_pointer __cudampi__getnextchunkindex(long long *globalcounter, long long max) { 
  return __cudampi__getnextchunkindex_enableddevices(globalcounter, __cudampi__getCurrentBatchSize(), max); 
}

__cudampi__batch_pointer __cudampi__getnextchunkindex_enableddevices(long long *globalcounter, unsigned long batchsize, long long max) {
  // for a given thread return the next available data chunk
  // max is the vector size
  __cudampi__batch_pointer batch_pointer = {max, 0};
  int deviceenabled;

  omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  deviceenabled = __cudampi__deviceenabled[__cudampi__currentDevice];
  omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

  if (deviceenabled == 1)
  {
    #pragma omp critical
    {
      if(*globalcounter < max)
      {
        batch_pointer.start = *globalcounter;
        (*globalcounter) += batchsize;
      }
    }

    if (batch_pointer.start < max)
    {
      batch_pointer.n_elements = (((batch_pointer.start + batchsize) > max )? max - batch_pointer.start : batchsize);
      __cudampi__batches_sent[omp_get_thread_num()] += 1;
      __cudampi__data_points_sent[omp_get_thread_num()] += batchsize;
    }
  }

  return batch_pointer;
}

__cudampi__batch_pointer __cudampi__getnextchunkindex_alldevices(long long *globalcounter, unsigned long batchsize, long long max) {
  // for a given thread (GPU) return the next available data chunk
  // max is the vector size
  __cudampi__batch_pointer batch_pointer = {max, 0};

  #pragma omp critical
  {
    if(*globalcounter < max)
    {
      batch_pointer.start = *globalcounter;
      (*globalcounter) += batchsize;
    }
  }

    if (batch_pointer.start < max)
    {
      batch_pointer.n_elements = (((batch_pointer.start + batchsize) > max )? max - batch_pointer.start : batchsize);
      __cudampi__batches_sent[omp_get_thread_num()] += 1;
      __cudampi__data_points_sent[omp_get_thread_num()] += batchsize;
    }

  return batch_pointer;
}

int __cudampi__isdeviceenabled(int deviceid) {
  int val;

  #pragma omp atomic read
  val = __cudampi__deviceenabled[__cudampi__currentDevice];

  return val;
}

cudaError_t __cudampi__getDeviceCount(int *count) {
  *count =  __cudampi_totaldevicecount;
  return cudaSuccess;
}

cudaError_t __cudampi__cudaGetDeviceCount(int *count) {

  *count = __cudampi_totalgpudevicecount;
  return cudaSuccess;
}

cudaError_t __cudampi__cpuGetDeviceCount(int *count) {

  *count = __cudampi_totalcpudevicecount;
  return cudaSuccess;
}

void __cudampi__initializeMPI(int argc, char **argv) {

  int mtsprovided;
  int i;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mtsprovided);

  if (mtsprovided != MPI_THREAD_MULTIPLE) {
    log_message(LOG_ERROR,"\nNo support for MPI_THREAD_MULTIPLE mode.\n");
    fflush(stdout);
    exit(-1);
  }

  /* Default values */
  __cudampi__arguments.cpu_enabled = 1;
  __cudampi__arguments.number_of_streams = 1;
  __cudampi__arguments.batch_size = 0;
  __cudampi__arguments.powercap = 0;
  __cudampi__arguments.cpu_power_scaling = 0.0;
  __cudampi__arguments.cpu_batch_scaling_factor = 0;
  __cudampi__arguments.use_dynamic_scaling = 1;

  /* Parse our arguments; every option seen by parse_opt will be reflected in arguments. */
  argp_parse(&argp, argc, argv, 0, 0, &__cudampi__arguments);

  if (__cudampi__arguments.batch_size == 0) {
    log_message(LOG_ERROR, "Failed to initialize cudampi - batch_size argument is required");
    exit(-1);
  }

  /* Print parsed arguments using log_message with LOG_INFO level */
  log_message(LOG_INFO, "CPU Enabled                                : %d",   __cudampi__arguments.cpu_enabled);
  log_message(LOG_INFO, "Number of Streams                          : %d",   __cudampi__arguments.number_of_streams);
  log_message(LOG_INFO, "Batch Size                                 : %d",   __cudampi__arguments.batch_size);
  log_message(LOG_INFO, "Power Cap                                  : %d",   __cudampi__arguments.powercap);
  log_message(LOG_INFO, "CPU Power Scaling                          : %f",   __cudampi__arguments.cpu_power_scaling);
  log_message(LOG_INFO, "Initial Cpu Batch Size Scaling Factor      : %d",   __cudampi__arguments.cpu_batch_scaling_factor);
  log_message(LOG_INFO, "Dynamic CPU Batch Size Scaling Enabled     : %d",   __cudampi__arguments.use_dynamic_scaling);

  __cudampi__dyanmicCpuBatchSizeScalingEnabled = __cudampi__arguments.use_dynamic_scaling;
  __cudampi__cpu_enabled = __cudampi__arguments.cpu_enabled;
  __cudampi__default_batch_size = __cudampi__arguments.batch_size;

  __cudampi__cpu_power_scaling = __cudampi__arguments.cpu_power_scaling;
  
  if (__cudampi__cpu_enabled == 0)
  {
    log_message(LOG_INFO, "Cpu disabled. Setting CPU power scaling to 0.0");
    __cudampi__cpu_power_scaling = 0.0;
  } else if (__cudampi__cpu_power_scaling < 0.01 || __cudampi__cpu_power_scaling >= 1.0)
  {
    // Check if it's within bounds <0.01 (just some small number); 1.0>
    log_message(LOG_INFO, "Setting CPU power scaling to 1.0");
    __cudampi__cpu_power_scaling = 1.0;
  }

  if (__cudampi__arguments.cpu_batch_scaling_factor == 0)
  {
    __cudampi__cpu_batch_size = __cudampi__default_batch_size;
  }
  else if (__cudampi__cpu_enabled)
  {
    __cudampi__cpu_batch_size = (unsigned long)(((double)__cudampi__default_batch_size) / ((double)(__cudampi__arguments.cpu_batch_scaling_factor)));
    log_message(LOG_INFO, "Scaled CPU batch size down to %ld from %ld (factor of %d)",
                __cudampi__cpu_batch_size, __cudampi__default_batch_size, __cudampi__arguments.cpu_batch_scaling_factor);
  }

  // initialize NVML for local GPU
  nvmlReturn_t nvmlResult = nvmlInit();
  if (nvmlResult != NVML_SUCCESS) {
      log_message(LOG_ERROR, "nvmlInit failed: %s\n", nvmlErrorString(nvmlResult));
      exit(-1);
  }

  if (__cudampi__arguments.powercap > 0) {
    log_message(LOG_INFO, "\nSetting power limit=%d\n", __cudampi__arguments.powercap);
    __cudampi__setglobalpowerlimit(__cudampi__arguments.powercap);
  }

  // fetch information about the rank and number of processes

  MPI_Comm_size(MPI_COMM_WORLD, &__cudampi__MPIproccount);
  MPI_Comm_rank(MPI_COMM_WORLD, &__cudampi__myrank);

  // we assume that there are as many nodes with GPUs as the number of processes started

  __cudampi__GPUcountspernode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__GPUcountspernode) {
    log_message(LOG_ERROR,"\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  __cudampi__freeThreadsPerNode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__freeThreadsPerNode) {
    log_message(LOG_ERROR,"\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  // initialize the array -- for simplicity first try to use all available GPUs in all nodes -- query the nodes

  // each process first checks its own device count
  if (cudaSuccess != cudaGetDeviceCount(&__cudampi__localGpuDeviceCount)) {
    log_message(LOG_ERROR,"Error invoking cudaGetDeviceCount()");
    fflush(stdout);
    exit(-1);
  }

  assert(__cudampi__localGpuDeviceCount < MAX_GPU_PER_NODE);

  MPI_Bcast(&__cudampi__cpu_enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&__cudampi__cpu_power_scaling, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Allgather(&__cudampi__localGpuDeviceCount, 1, MPI_INT, __cudampi__GPUcountspernode, 1, MPI_INT, MPI_COMM_WORLD);

  // Master does not use local free threads for computations
  __cudampi__localFreeThreadCount = 0;

  MPI_Allgather(&__cudampi__localFreeThreadCount, 1, MPI_INT, __cudampi__freeThreadsPerNode, 1, MPI_INT, MPI_COMM_WORLD);

  // check if there is a configuration file
  FILE *filep = fopen("__cudampi.conf", "r");

  if (filep != NULL) {
    char line[255];
    int index, val;
    while (NULL != fgets(line, 255, filep)) {
      sscanf(line, "%d:%d", &index, &val);
      __cudampi__GPUcountspernode[index] = val;
    }

    fclose(filep);
  }

  // compute the total device count
  __cudampi_totalgpudevicecount = 0;
  for (i = 0; i < __cudampi__MPIproccount; i++) {
    __cudampi_totalgpudevicecount += __cudampi__GPUcountspernode[i];
    log_message(LOG_INFO,"\nOn node %d using %d GPUs.", i, __cudampi__GPUcountspernode[i]);
    if (__cudampi__freeThreadsPerNode[i] > 0 ) {
      __cudampi_totalcpudevicecount ++;
    }
    log_message(LOG_INFO,"\nOn node %d using %d CPU threads.\n", i, __cudampi__freeThreadsPerNode[i]);
  }

  __cudampi_totaldevicecount = __cudampi_totalcpudevicecount + __cudampi_totalgpudevicecount;

  fflush(stdout);

  // now compute proper indexes

  __cudampi_targetGPUfordevice = (int *)malloc(__cudampi_totaldevicecount * sizeof(int));
  if (!__cudampi_targetGPUfordevice) {
    log_message(LOG_ERROR,"\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  __cudampi_targetMPIrankfordevice = (int *)malloc(__cudampi_totaldevicecount * sizeof(int));
  if (!__cudampi_targetMPIrankfordevice) {
    log_message(LOG_ERROR,"\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  int currentrank = 0;
  int currentGPU = 0;
  // now initialize values device by device
  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    __cudampi_targetGPUfordevice[i] = currentGPU;
    __cudampi_targetMPIrankfordevice[i] = currentrank;

    __cudampi__devicepower[i] = -1; // initial value
    __cudampi__deviceenabled[i] = 1;

    currentGPU++;

    if (currentGPU == __cudampi__GPUcountspernode[currentrank]) {
      // reset the GPU id and increase the rank
      currentGPU = 0;
      currentrank++;
    }
  }

  currentrank = 0;
  for (i = __cudampi_totalgpudevicecount; i < __cudampi_totaldevicecount; i++) {
    __cudampi_targetGPUfordevice[i] = -1;
    __cudampi__devicepower[i] = -1; // initial value
    __cudampi__deviceenabled[i] = 1;
    while (__cudampi__freeThreadsPerNode[currentrank] <= 0)
    {
      // skip all nodes with no free threads;
      currentrank ++;
    }
    __cudampi_targetMPIrankfordevice[i] = currentrank;
    currentrank ++;
  }
  // initialize current device id to 0 although various threads are expected to have various GPU ids

  for (int i = 0; i < __CUDAMPI_MAX_THREAD_COUNT; i++) {
    __cudampi__currentdevice[i] = 0;

    // initialize locks as well
    omp_init_lock(&(__cudampi__devicelocks[i]));
  }

  omp_init_lock(&deviceselectionlock);

  __cudampi__amimanager[0] = 1;

  MPI_Bcast(&__cudampi_totaldevicecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(__cudampi_targetMPIrankfordevice, __cudampi_totaldevicecount, MPI_INT, 0, MPI_COMM_WORLD);

  // set up communicators - there should be a communicator for each target GPU, each of which is shared by process 0 and process on which the target GPU resides, this is obviously not needed for local GPUs

  __cudampi__communicators = (MPI_Comm *)malloc(sizeof(MPI_Comm) * __cudampi_totaldevicecount);
  if (!__cudampi__communicators) {
    log_message(LOG_ERROR,"\nNot enough memory for communicators");
    exit(-1); // we could exit in a nicer way! TBD
  }

  for (int i = __cudampi__GPUcountspernode[0]; i < __cudampi_totaldevicecount; i++) { // disregard local GPUs since we do not need communicators for them -- communication will be by direct invocations

    int ranks[2] = {0, __cudampi_targetMPIrankfordevice[i]}; // group and communicator between process 0 and the process of the target GPU/device

    MPI_Group groupall;
    MPI_Comm_group(MPI_COMM_WORLD, &groupall);

    // Keep only process 0 and the process handling the given GPU
    MPI_Group tempgroup;
    MPI_Group_incl(groupall, 2, ranks, &tempgroup);

    MPI_Comm_create(MPI_COMM_WORLD, tempgroup, &(__cudampi__communicators[i]));
  }

  for (int i = 0; i < __cudampi_totaldevicecount; i++ ) {
    TAILQ_INIT(&(__cudampi__memcpy_queues[i]));
  }

  for (int i = 0; i < MAX_THREADS; i++) {
    omp_init_lock(&cpuEnergyLock[i]);
  }
}

void __cudampi__terminateMPI() {

  for (int i = 0; i < __cudampi_totaldevicecount;i++){
    log_message(LOG_INFO, "Batches sent by thread %d: %ld", i, __cudampi__batches_sent[i]);
    log_message(LOG_INFO, "Data points processed by thread %d: %lld", i, __cudampi__data_points_sent[i]);
  }

  // finalize the other nodes -> shut down threads responsible for remote GPUs

  for (int i = __cudampi__localGpuDeviceCount; i < __cudampi_totaldevicecount; i++) {
    MPI_Send(NULL, 0, MPI_CHAR, 1, __cudampi__CUDAMPIFINALIZE, __cudampi__communicators[i]);
  }

  nvmlShutdown();

  log_message(LOG_WARN, "Terminating CUDAMPILIB, Total energy used %lf J", __cudampi__totalEnergyUsed);

  
  for (int i = 0; i < MAX_THREADS; i++) {
    omp_destroy_lock(&cpuEnergyLock[i]);
  }

  MPI_Finalize();
}

int __cudampi__gettargetGPU(int device) {
  // gets target GPU id

  return __cudampi_targetGPUfordevice[device];
  //  return device%__cudampi__GPUcountpernode;
}

int __cudampi__gettargetMPIrank(int device) {
  // gets target MPI rank based on local GPU id

  return __cudampi_targetMPIrankfordevice[device];
  //  return device/__cudampi__GPUcountpernode;
}

cudaError_t __cudampi__cudaMalloc(void **devPtr, size_t size) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaMalloc(devPtr, size);
  } else { // allocate remotely

    // we then return the actual pointer from another node -- it is used only on that node

    // request allocation on the other node

    unsigned long sdata = size; // how many bytes to allocate on the GPU

    MPI_Send((void *)(&sdata), 1, MPI_UNSIGNED_LONG, 1, __cudampi__CUDAMPIMALLOCREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(void *) + sizeof(cudaError_t);
    // receive confirmation with the actual pointer
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIMALLOCRESP, __cudampi__currentCommunicator, NULL);

    *devPtr = *((void **)rdata);

    return ((cudaError_t)(rdata + sizeof(void *)));
  }
}

cudaError_t __cudampi__cpuMalloc(void **devPtr, size_t size) {
  // allocate remotely

  // we then return the actual pointer from another node -- it is used only on that node

  // request allocation on the other node

  unsigned long sdata = size; // how many bytes to allocate on the CPU

  MPI_Send((void *)(&sdata), 1, MPI_UNSIGNED_LONG, 1, __cudampi__CPUMALLOCREQ, __cudampi__currentCommunicator);

  int rsize = sizeof(void *) + sizeof(cudaError_t);
  // receive confirmation with the actual pointer
  unsigned char rdata[rsize];

  MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUMALLOCRESP, __cudampi__currentCommunicator, NULL);

  *devPtr = *((void **)rdata);

  return ((cudaError_t)(rdata + sizeof(void *)));
}

cudaError_t __cudampi__malloc(void **devPtr, size_t size) {
  if (__cudampi__isCpu()) {
    return __cudampi__cpuMalloc(devPtr, size);
  }
  // else
  return __cudampi__cudaMalloc(devPtr, size);
}


cudaError_t __cudampi__cudaFree(void *devPtr) {
  // as for cudaMalloc but just free

  if (__cudampi_isLocalGpu) { // run locally
    return cudaFree(devPtr);
  } else { // allocate remotely
    int ssize = sizeof(void *);
    unsigned char sdata[ssize];

    *((void **)sdata) = devPtr;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIFREEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIFREERESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cpuFree(void *devPtr) {
  // allocate remotely
  // data for sending (devPtr pointer address)
  int ssize = sizeof(void *);
  unsigned char sdata[ssize];

  *((void **)sdata) = devPtr;

  MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUFREEREQ, __cudampi__currentCommunicator);

  int rsize = sizeof(cudaError_t);
  unsigned char rdata[rsize];

  MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUFREERESP, __cudampi__currentCommunicator, NULL);

  return *((cudaError_t *)rdata);
}

cudaError_t __cudampi__free(void *devPtr) {
  if (__cudampi__isCpu()) {
    return __cudampi__cpuFree(devPtr);
  }
  // else
  return __cudampi__cudaFree(devPtr);
}

cudaError_t __cudampi__cudaDeviceSynchronize(void)
{
  return __cudampi__deviceSynchronize();
}

cudaError_t __cudampi__deviceSynchronize(void) {

  cudaError_t retVal;
  static int selecteddevices = 0; // only updated by thread 0
  int amimanager;                 // if the current thread is manager for device selection
  float energy = -1, power = -1;

  // if ((powermeasurecounter[omp_get_thread_num()]%10)==4) {

  omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  amimanager = __cudampi__amimanager[__cudampi__currentDevice];
  omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

  if (amimanager) {
    if (__cudampi__isglobalpowerlimitset) {
      if (!selecteddevices) {
        selecteddevices = __cudampi__selectdevicesforpowerlimit_greedy();
      } else {

        // adjust if needed
        float power;
        power = __cudampi__gettotalpowerofselecteddevices();
        if (power != (-1)) {
          if (power > __cudampi__globalpowerlimit) {
            log_message(LOG_DEBUG,"\ntotal power=%f limit=%f, adjusting", power, __cudampi__globalpowerlimit);
            fflush(stdout);
            __cudampi__selectdevicesforpowerlimit_greedy();
          }
        }
      }
    }
  }

  if (__cudampi_isLocalGpu) { // run GPU synchronization locally

    // now get power measurement - this should be OK as we assume that computations might be taking place
    if (1) {
      cudaError_t error = cudaErrorUnknown;
      error = getCpuEnergyUsed(&cpuLastEnergyMeasured[omp_get_thread_num()], &energy);
      energy /= __cudampi__localGpuDeviceCount;
      if (error != cudaSuccess) {
        energy = -1;
      }
      power = getGPUpower(__cudampi__currentDevice);
      log_message(LOG_DEBUG, "Got local GPU power %f and CPU energy for this GPU %f", power, energy);
    }

    retVal = cudaDeviceSynchronize();
  } else { // run synchronization remotely
    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    int sdata = 1; // if 0 then means do not measure power, if 1 do measure on the slave side

    int rsize = sizeof(cudaError_t) + sizeof(float) + sizeof(float);
    unsigned char rdata[rsize];
    if (__cudampi__isCpu())
    {
      rsize = sizeof(cudaError_t) + sizeof(float);

      MPI_Send(&sdata, 1, MPI_INT, 1, __cudampi__CUDAMPICPUDEVICESYNCHRONIZEREQ, __cudampi__currentCommunicator);

      // receive an error message and a float representing power consumption


      MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPICPUDEVICESYNCHRONIZERESP, __cudampi__currentCommunicator, NULL);

      // decode and store power consumption for the device

      energy = *((float *)(rdata + sizeof(cudaError_t)));
      log_message(LOG_DEBUG, "Got CPU energy %f", energy);
    }
    else
    {
      MPI_Send(&sdata, 1, MPI_INT, 1, __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ, __cudampi__currentCommunicator);

      // receive an error message and a float representing power consumption for GPU and float with energy used by CPU (scaled down)

      MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIDEVICESYNCHRONIZERESP, __cudampi__currentCommunicator, NULL);

      // decode and store power consumption for the device

      power = *((float *)(rdata + sizeof(cudaError_t)));
      energy = *((float *)(rdata + sizeof(cudaError_t) + sizeof(float)));
      log_message(LOG_DEBUG, "Got remote GPU power %f and CPU energy for this GPU %f", power, energy);
    }
    process_queue();

    retVal = ((cudaError_t)rdata);
  }

  powermeasurecounter[omp_get_thread_num()]++;

  // record time
  if (__cudampi__timemeasured[__cudampi__currentDevice]) {
    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
    struct timeval elapsed_time;
    gettimeofday(&(__cudampi__timestop[__cudampi__currentDevice]), NULL);

    // Compute time elapsed
    elapsed_time.tv_sec = __cudampi__timestop[__cudampi__currentDevice].tv_sec - __cudampi__timestart[__cudampi__currentDevice].tv_sec; 
    elapsed_time.tv_usec = __cudampi__timestop[__cudampi__currentDevice].tv_usec - __cudampi__timestart[__cudampi__currentDevice].tv_usec;

    __cudampi__timestart[__cudampi__currentDevice] = __cudampi__timestop[__cudampi__currentDevice];

    // Convert to seconds and microseconds
    __cudampi__time_us[__cudampi__currentDevice] = elapsed_time.tv_sec * 1000000 + elapsed_time.tv_usec;
    double time_in_seconds = __cudampi__time_us[__cudampi__currentDevice] / 1000000.0;

    // Count batches sent from last iteration
    unsigned long batches_sent = __cudampi__batches_sent[__cudampi__currentDevice] - __cudampi__last_batches_sent[__cudampi__currentDevice];
    unsigned long long data_points_sent = __cudampi__data_points_sent[__cudampi__currentDevice] - __cudampi__last_data_points_sent[__cudampi__currentDevice];
    __cudampi__last_batches_sent[__cudampi__currentDevice] = __cudampi__batches_sent[__cudampi__currentDevice];
    __cudampi__last_data_points_sent[__cudampi__currentDevice] = __cudampi__data_points_sent[__cudampi__currentDevice];

    double scaling_factor = 1;
    if (batches_sent > 0) {
      scaling_factor = ((double)data_points_sent) / ((double)(batches_sent * __cudampi__default_batch_size));
    }

    log_message(LOG_DEBUG, "Scaling factor for power capping = %lf", scaling_factor);

    // Divide time by scaling factor (which is smaller than 1) which means that
    // if batch size for that device was scaled down, time it would take to process full batch of data is calculated
    __cudampi__time_us[__cudampi__currentDevice] /= scaling_factor;

    // assert(energy != -1);
    if (__cudampi__isCpu() && (energy != -1)){
      power = energy / time_in_seconds;
    }

    if (!__cudampi__isCpu() && (energy != -1)) {
      power += (energy / time_in_seconds);
    }

    if (power != (-1)) {
      __cudampi__devicepower[__cudampi__currentDevice] = power;
      
      log_message(LOG_DEBUG, "Got power %f with time in seconds = %f", power, time_in_seconds);
      #pragma omp atomic
      __cudampi__totalEnergyUsed += power * time_in_seconds;
    }
    
    // Check if CPU batch size scaling was already done
    int scalingDone;
    #pragma omp atomic read
    scalingDone = __cudampi__cpuBatchSizeScalingDone;

    if (__cudampi__dyanmicCpuBatchSizeScalingEnabled && (!scalingDone)) {
      __cudampi__finishDeviceStatsMeasurement(time_in_seconds);
      __cudampi__scaleCpuBatchSize();
    }

    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  } else {
    __cudampi__timemeasured[__cudampi__currentDevice] = 1;
    gettimeofday(&(__cudampi__timestart[__cudampi__currentDevice]), NULL);
  }

  return retVal;
}

cudaError_t __cudampi__cudaSetDevice(int device) {

  __cudampi__currentDevice = device; // set it for the current thread

  if (__cudampi_isLocalGpu) { // run locally
    return cudaSetDevice(device);
  } else { // set device remotely
    int sdata = __cudampi__gettargetGPU(__cudampi__currentDevice); // compute the target GPU id

    MPI_Send(&sdata, 1, MPI_INT, 1, __cudampi__CUDAMPISETDEVICEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPISETDEVICERESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);
  }
}

int __cudampi__isCpu()
{
  // TODO ?
  return __cudampi__currentDevice  >= __cudampi_totalgpudevicecount;
}

cudaError_t __cudampi__setDevice(int device) {
  __cudampi__currentDevice = device; // set it for the current thread
  if (!__cudampi__isCpu()) {
    return __cudampi__cudaSetDevice(device);
  }
  // else
  return cudaSuccess;
}

cudaError_t __cudampi__cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaMemcpy(dst, src, count, kind);
  } else if (kind == cudaMemcpyHostToDevice) {

    size_t ssize = sizeof(void *) + count;
    unsigned char *sdata = malloc(ssize);

    *((void **)sdata) = dst;
    memcpy(sdata + sizeof(void *), src, count); // copy input data

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIHOSTTODEVICEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIHOSTTODEVICERESP, __cudampi__currentCommunicator, NULL);

    free(sdata);
    return ((cudaError_t)rdata);

  } else if (kind == cudaMemcpyDeviceToHost) {

    size_t ssize = sizeof(void *) + sizeof(unsigned long);
    unsigned char sdata[ssize];

    *((void **)sdata) = (void *)src;
    *((unsigned long *)(sdata + sizeof(void *))) = count; // how many bytes we want to get from a GPU

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIDEVICETOHOSTREQ, __cudampi__currentCommunicator);

    size_t rsize = sizeof(cudaError_t) + count;
    unsigned char* rdata = malloc(rsize);

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIDEVICETOHOSTRESP, __cudampi__currentCommunicator, NULL);

    memcpy(dst, rdata + sizeof(cudaError_t), count);

    free(rdata);
    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cpuMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  // run remotely
  if (kind == cudaMemcpyHostToDevice) {
    size_t ssize = sizeof(void *) + count;
    unsigned char* sdata = malloc(ssize);

    *((void **)sdata) = dst;
    memcpy(sdata + sizeof(void *), src, count); // copy input data

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUHOSTTODEVICEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUHOSTTODEVICERESP, __cudampi__currentCommunicator, NULL);

    free(sdata);
    return ((cudaError_t)rdata);

  } else if (kind == cudaMemcpyDeviceToHost) {

    size_t ssize = sizeof(void *) + sizeof(unsigned long);
    unsigned char sdata[ssize];

    *((void **)sdata) = (void *)src;
    *((unsigned long *)(sdata + sizeof(void *))) = count; // how many bytes we want to get from a GPU

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUDEVICETOHOSTREQ, __cudampi__currentCommunicator);

    size_t rsize = sizeof(cudaError_t) + count;
    unsigned char* rdata = malloc(rsize);

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUDEVICETOHOSTRESP, __cudampi__currentCommunicator, NULL);

    memcpy(dst, rdata + sizeof(cudaError_t), count);

    free(rdata);
    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {

  int counter = getMsgCounter();
  if (__cudampi_isLocalGpu) { // run locally
    return cudaMemcpyAsync(dst, src, count, kind, stream);
  } else if (kind == cudaMemcpyHostToDevice) {

    size_t ssize = sizeof(void *) + sizeof(cudaStream_t) + sizeof(int) + count;
    unsigned char *sdata = malloc(ssize);

    *((void **)sdata) = dst;
    *((cudaStream_t *)(sdata + sizeof(void *))) = stream;
    *((int *)(sdata + sizeof(void *) + sizeof(cudaStream_t))) = counter;
    memcpy(sdata + sizeof(void *) + sizeof(cudaStream_t) + sizeof(int), src, count); // copy input data

    waitForAsyncSendResponse(counter);

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ, __cudampi__currentCommunicator);
    free(sdata);
    return cudaSuccess;

  } else if (kind == cudaMemcpyDeviceToHost) {

    size_t ssize = sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t) + sizeof(int);
    unsigned char sdata[ssize];

    *((void **)sdata) = (void *)src;
    *((unsigned long *)(sdata + sizeof(void *))) = count; // how many bytes we want to get from a GPU
    *((cudaStream_t *)(sdata + sizeof(void *) + sizeof(unsigned long))) = stream;
    *((int *)(sdata + sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t))) = counter;

    initiateAsyncRecv(dst, count, counter);

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ, __cudampi__currentCommunicator);

    return cudaSuccess;
  }
}

cudaError_t __cudampi__cpuMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  // run remotely
  int counter = getMsgCounter();
  if (kind == cudaMemcpyHostToDevice) {
    // First just send the request with number of bytes
    size_t ssize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long) + sizeof(int);
    unsigned char sdata[ssize];
    *((void **)sdata) = dst;
    *((unsigned long*)(sdata + sizeof(void*))) = count;
    *((unsigned long *)(sdata + sizeof(void *) + sizeof(unsigned long))) = (unsigned long)stream;
    *((int *)(sdata + sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long))) = counter;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUHOSTTODEVICEREQASYNC, __cudampi__currentCommunicator);

    // Then send the data asynchronously 
    initiateAsyncSendCpu(src, count, counter);

    return cudaSuccess;

  } else if (kind == cudaMemcpyDeviceToHost) {

    size_t ssize = sizeof(void *) + sizeof(unsigned long)  + sizeof(unsigned long)  + sizeof(int);
    unsigned char sdata[ssize];

    *((void **)sdata) = (void *)src;
    *((unsigned long *)(sdata + sizeof(void *))) = count;
    *((unsigned long *)(sdata + sizeof(void *) + sizeof(unsigned long))) = (unsigned long)stream;
    *((int *)(sdata + sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long))) = counter;

    // Start receiveing data asynchronously
    initiateAsyncRecv(dst, count, counter);

    // Send the request
    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUDEVICETOHOSTREQASYNC, __cudampi__currentCommunicator);

    return cudaSuccess;
  }
}

void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long id);

void __cudampi__cudaKernelInStream(void *devPtr, unsigned long batchsize, cudaStream_t stream, unsigned long long id) {
  __cudampi__recordBatchSizeForDeviceStats(batchsize);

  if (__cudampi_isLocalGpu) { // run locally
    initializeCpuEnergyMeasurement(isInitialCpuEnergyMeasured, cpuEnergyLock, cpuLastEnergyMeasured);
    launchkernelinstream(devPtr, batchsize, stream, id);
  } else { // launch remotely

    size_t ssize = sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t) + sizeof(unsigned long long);
    unsigned char sdata[ssize];

    *((void **)sdata) = devPtr;
    *((unsigned long*)(sdata + sizeof(void *))) = batchsize;
    *((cudaStream_t *)(sdata + sizeof(void *) + sizeof(unsigned long))) = stream;
    *((unsigned long long *)(sdata + sizeof(void *) + sizeof(cudaStream_t))) = id;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ, __cudampi__currentCommunicator);
    // No need to wait for response since all kernels return void
  }
}

void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id);  // extern from .cu

void __cudampi__cudaKernel(void *devPtr, unsigned long batchsize, unsigned long long id) {
  __cudampi__recordBatchSizeForDeviceStats(batchsize);

  if (__cudampi_isLocalGpu) { // run locally
    initializeCpuEnergyMeasurement(isInitialCpuEnergyMeasured, cpuEnergyLock, cpuLastEnergyMeasured);
    launchkernel(devPtr, batchsize, id);
  } else { // launch remotely

    size_t ssize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long long);
    unsigned char sdata[ssize];

    *((void **)sdata) = devPtr;
    *((unsigned long*)(sdata + sizeof(void *))) = batchsize;
    *((unsigned long long*)(sdata + sizeof(void *) + sizeof(unsigned long))) = id;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPILAUNCHCUDAKERNELREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPILAUNCHCUDAKERNELRESP, __cudampi__currentCommunicator, NULL);
  }
}

void __cudampi__cpuKernelInStream(void *devPtr, unsigned long batchsize, cudaStream_t stream, unsigned long long id){
  __cudampi__recordBatchSizeForDeviceStats(batchsize);

  // launch remotely - since master does not use local threads for computations
  size_t ssize = sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t) + sizeof(unsigned long long);
  unsigned char sdata[ssize];

  *((void **)sdata) = devPtr;
  *((unsigned long*)(sdata + sizeof(void *))) = batchsize;
  *((cudaStream_t *)(sdata + sizeof(void *) + sizeof(unsigned long))) = stream;
  *((unsigned long long *)(sdata + sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t))) = id;

  MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPULAUNCHKERNELREQ, __cudampi__currentCommunicator);
  // No need to wait for response since all kernels return void
}

void __cudampi__cpuKernel(void *devPtr, unsigned long batchsize, unsigned long long id) {
  __cudampi__cpuKernelInStream(devPtr, batchsize, NULL, id);
}

cudaError_t __cudampi__cudaStreamCreate(cudaStream_t *pStream) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaStreamCreate(pStream);
  } else { // create a stream remotely

    // we then return the actual pointer from another node -- it is used only on that node

    // send an empty message -- there is no need for input data
    MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, 1, __cudampi__CUDAMPISTREAMCREATEREQ, __cudampi__currentCommunicator);

    // get a pointer to the newly created stream + an error message
    int rsize = sizeof(cudaStream_t) + sizeof(cudaError_t);
    // receive confirmation with the actual pointer
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPISTREAMCREATERESP, __cudampi__currentCommunicator, NULL);

    *pStream = *((cudaStream_t *)rdata);

    return ((cudaError_t)(rdata + sizeof(void *)));
  }
}

cudaError_t __cudampi__cudaStreamDestroy(cudaStream_t stream) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaStreamDestroy(stream);
  } else { // destroy remotely

    int ssize = sizeof(cudaStream_t);
    unsigned char sdata[ssize];

    *((cudaStream_t *)sdata) = stream;

    MPI_Send((unsigned char *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPISTREAMDESTROYREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CUDAMPISTREAMDESTROYRESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cpuStreamCreate(cudaStream_t *pStream) {
  // create a stream remotely
  // we then return the actual pointer from another node -- it is used only on that node

  // send an empty message -- there is no need for input data
  MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, 1, __cudampi__CPUSTREAMCREATEREQ, __cudampi__currentCommunicator);

  // get a pointer to the newly created stream + an error message
  int rsize = sizeof(unsigned long) + sizeof(cudaError_t);
  // receive confirmation with the actual pointer
  unsigned char rdata[rsize];

  MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUSTREAMCREATERESP, __cudampi__currentCommunicator, NULL);

  *pStream = (cudaStream_t)*((unsigned long *)rdata);

  return ((cudaError_t)(rdata + sizeof(void *)));
}

cudaError_t __cudampi__cpuStreamDestroy(cudaStream_t stream) {
  // destroy remotely

  int ssize = sizeof(unsigned long);
  unsigned char sdata[ssize];

  *((unsigned long *)sdata) = (unsigned long)stream;

  MPI_Send((unsigned char *)sdata, ssize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUSTREAMDESTROYREQ, __cudampi__currentCommunicator);

  int rsize = sizeof(cudaError_t);
  unsigned char rdata[rsize];

  MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1, __cudampi__CPUSTREAMDESTROYRESP, __cudampi__currentCommunicator, NULL);

  return ((cudaError_t)rdata);
}

cudaError_t __cudampi__streamCreate(cudaStream_t *stream) {
  if (__cudampi__isCpu()) {
    return __cudampi__cpuStreamCreate(stream);;
  }
  // else
  return __cudampi__cudaStreamCreate(stream);
}

cudaError_t __cudampi__streamDestroy(cudaStream_t stream) {
  if (__cudampi__isCpu()) {
    return __cudampi__cpuStreamDestroy(stream);
  }
  // else
  return __cudampi__cudaStreamDestroy(stream);
}

cudaError_t __cudampi__memcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (__cudampi__isCpu())
  {
    return __cudampi__cpuMemcpyAsync(dst, src, count, kind, stream);
  }
  // else
  return __cudampi__cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t __cudampi__memcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  if (__cudampi__isCpu())
  {
    return __cudampi__cpuMemcpy(dst, src, count, kind);
  }
  // else
  return __cudampi__cudaMemcpy(dst, src, count, kind);
}

void __cudampi__kernelInStream(void *devPtr, cudaStream_t stream, unsigned long long id) {
  if (__cudampi__isCpu())
  {
    return __cudampi__cpuKernelInStream(devPtr, __cudampi__getCurrentBatchSize(), stream, id);
  }
  // else
  return __cudampi__cudaKernelInStream(devPtr, __cudampi__getCurrentBatchSize(), stream, id);
}

void __cudampi__kernel(void *devPtr, unsigned long long id) {
  if (__cudampi__isCpu())
  {
    return __cudampi__cpuKernel(devPtr, __cudampi__getCurrentBatchSize(), id);
  }
  // else
  return __cudampi__cudaKernel(devPtr, __cudampi__getCurrentBatchSize(), id);
}