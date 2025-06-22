/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/queue.h>
#include <nvml.h>
#include <assert.h>

#include "cudampi.h"
#include "cudampicommon.h"

// number of stream dedicated for sending GPU responses to master
#define CPU_STREAM_FOR_GPU_RESPONSES CPU_STREAMS_SUPPORTED + omp_get_thread_num()
// 4 GB per GPU seems reasonable
#define INITIAL_GPU_BUFFER_SIZE 4 * 1024 * 1024 * 1024UL

#define ENABLE_LOGGING
#define MPI_LOGGING
#include "logger.h"

#define ENABLE_CHECKING_MESSAGE_TYPE

int __cudampi__MPIproccount;
int __cudampi__myrank;
int terminated[ALL_CPU_STREAMS] = {0};

int debugTaskCounter = 0;

int *__cudampi_targetMPIrankfordevice; // MPI rank for device number (global)
int *__cudampi__GPUcountspernode;
int *__cudampi__freeThreadsPerNode;

MPI_Comm *__cudampi__communicators;

int __cudampi__totaldevicecount = 0; // how many GPUs in total (on all considered nodes)
int __cudampi__localGpuDeviceCount = 1;
int __cudampi__localFreeThreadCount = 0;

int __cudampi__cpu_enabled;
float __cudampi__cpu_power_scaling;

unsigned long cpuStreamsValid[CPU_STREAMS_SUPPORTED];

typedef struct task_queue_entry {
    TAILQ_ENTRY(task_queue_entry) entries; // Use sys/queue.h macros for queue entries
    void (*task_func)(void *);
    void *arg;
    int id;
    int dep_var;   // Dependency variable for OpenMP
    int scheduled;    // Flag to indicate if the task has been scheduled
} task_queue_entry_t;

TAILQ_HEAD(task_queue_head, task_queue_entry) task_queues[ALL_CPU_STREAMS];
omp_lock_t queue_locks[ALL_CPU_STREAMS];

// CPU task launcher thread will wait on this lock until there are tasks available
// When there are no tasks in the queue, the lock is set
// When new task is being added to the queue, lock is being unset
omp_lock_t task_available_locks[ALL_CPU_STREAMS];
omp_lock_t synchronize_locks[ALL_CPU_STREAMS];

int scheduledTasksInStream[ALL_CPU_STREAMS];

// Below deinitions are for CPU energy measurements 
// (for measuring power consumption of CPU computation and managing the GPU)
float cpuLastEnergyMeasured[MAX_THREADS] = {0.0};
omp_lock_t cpuEnergyLock[MAX_THREADS];
int isInitialCpuEnergyMeasured[MAX_THREADS] = {0};

void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id);
void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long id);
void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long id);

typedef struct {
  unsigned char* buffer;
  unsigned long long  size;
  unsigned long long  overflow;
  unsigned long long pointer;
  unsigned long long  allocatedCount;
  omp_lock_t lock;
} globalGpuMemcpyBuffer;

typedef enum {
  ALLOCATED_IN_GLOBAL_BUFFER,
  ALLOCATED_SEPARATELY
} bufferAllocationType;

typedef struct {
  unsigned char * buffer;
  bufferAllocationType allocation;
  globalGpuMemcpyBuffer* globalPtr;
} allocatedGpuMemcpyBuffer;

typedef struct {
  void *devPtr;
  size_t count;
  int tag;
  MPI_Comm* comm;
} cpu_host_to_device_args_t;

typedef struct {
  cudaEvent_t event;
  int tag;
  MPI_Comm* comm;
  allocatedGpuMemcpyBuffer buffer;
} gpu_host_to_device_args_t;

typedef struct {
  void *devPtr;
  unsigned long count;
  int tag;
  MPI_Comm* comm;
} cpu_device_to_host_args_t;

typedef struct {
  cudaEvent_t event;
  allocatedGpuMemcpyBuffer buffer;
  unsigned long count;
  int tag;
  MPI_Comm* comm;
} gpu_device_to_host_args_t;

typedef struct {
  void *kernelArg;
  unsigned long batchSize;
  unsigned long long id;
} launch_kernel_args_t;

allocatedGpuMemcpyBuffer allocateGpuMemcpyBuffer (globalGpuMemcpyBuffer* global, unsigned long long count) {
  unsigned char* data;
  bufferAllocationType allocationType;

  omp_set_lock(&global->lock);

  if (global->pointer + count > global->size) {
    // Leave it as warning since the buffer should contain enoguh space in all example apps
    log_message(LOG_WARN, "Allocating separetely %lld when size = %lld and pointer = %lld\n", count, global->size, global->pointer);
    // Global buffer is not large enough to allocate that memory
    // Allocate it separately and increment overflow bytes in global buffer
    cudaError_t e = cudaHostAlloc((void**)&data, count, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        log_message(LOG_ERROR, "allocateGpuMemcpyBuffer: Error allocating host memory");
        data = NULL;
    }

    global->overflow += count;
    allocationType = ALLOCATED_SEPARATELY;
  }
  else {
    // Return pointer to global buffer
    data = global->buffer + global->pointer;
    // Move pointer by count
    global->pointer += count;
    global->allocatedCount += 1;
    allocationType = ALLOCATED_IN_GLOBAL_BUFFER;
  }

  omp_unset_lock(&global->lock);

  allocatedGpuMemcpyBuffer ret = {data, allocationType, global};
  return ret;
}

void freeGpuMemcpyBuffer(allocatedGpuMemcpyBuffer* allocatedBuffer) {
  switch (allocatedBuffer->allocation)
  {
  case ALLOCATED_SEPARATELY:
    cudaFreeHost(allocatedBuffer->buffer);
    break;
  case ALLOCATED_IN_GLOBAL_BUFFER:
    // Just decrement the count and leave the memory as is
    omp_set_lock(&allocatedBuffer->globalPtr->lock);
    allocatedBuffer->globalPtr->allocatedCount -= 1;
    omp_unset_lock(&allocatedBuffer->globalPtr->lock);
    break;
  default:
    log_message(LOG_ERROR, "Unknown allocation type");
    break;
  }
}

void updateGlobalGpuMemcpyBuffer(globalGpuMemcpyBuffer* global) {
  omp_set_lock(&global->lock);
  if (global->allocatedCount > 0) {
    log_message(LOG_ERROR, "Trying to reallocate gpu memcpy buffer while there are still %lld allocated entries !", global->allocatedCount);
  }

  if(global->overflow > 0) {
    // If there was overflow, enlarge the buffer and reallocate it
    log_message(LOG_WARN, "Enlarging GPU memcpy buffer");
    global->size += global->overflow;
    cudaFreeHost(global->buffer);
    cudaError_t e = cudaHostAlloc((void**)&global->buffer, global->size, cudaHostAllocDefault);

    if (e != cudaSuccess) {
        log_message(LOG_ERROR, "updateGlobalGpuMemcpyBuffer: Error allocating host memory");
    }
  }
  global->overflow = 0;
  global->pointer = 0;
  omp_unset_lock(&global->lock);
}

void initializeGlobalGpuMemcpyBuffer(globalGpuMemcpyBuffer* global, unsigned long count) {
  global->pointer = 0;
  global->allocatedCount = 0;
  global->overflow = 0;
  global->size = count;
  omp_init_lock(&global->lock);

  cudaError_t e = cudaHostAlloc((void**)&global->buffer, global->size, cudaHostAllocDefault);

  if (e != cudaSuccess) {
      log_message(LOG_ERROR, "initializeGlobalGpuMemcpyBuffer: Error allocating host memory");
  }
}

void freeGlobalGpuMemcpyBuffer(globalGpuMemcpyBuffer* global) {
  cudaFreeHost(global->buffer);
  omp_destroy_lock(&global->lock);
}

void cpuSynchronize()
{
  log_message(LOG_DEBUG, "Synchronizing CPU tasks");
  // Check if there are tasks waiting to be synchronized
  for (int i = 0; i < CPU_STREAMS_SUPPORTED; i++)
  {
    omp_set_lock(&synchronize_locks[i]);
    // Free the lock back
    omp_unset_lock(&synchronize_locks[i]);
  }
}

void cpuHostToDeviceTaskAsync(void* arg) {
  cudaError_t e = cudaErrorInvalidValue;
  cpu_host_to_device_args_t *args = (cpu_host_to_device_args_t*) arg;

  if (args->devPtr != NULL && args->count > 0){
    e = cudaSuccess;
    MPI_Recv((unsigned char *)args->devPtr, args->count, MPI_UNSIGNED_CHAR, 0, args->tag, *(args->comm), NULL);
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0,  args->tag + 1, *(args->comm));

  free(arg);
}

void cpuDeviceToHostTaskAsync(void* arg) {
  cudaError_t e = cudaErrorInvalidValue;
  cpu_device_to_host_args_t *args = (cpu_device_to_host_args_t*) arg;
  
  if (args->devPtr != NULL && args->count > 0){
    e = cudaSuccess;
    MPI_Send((unsigned char*)(args->devPtr), args->count , MPI_UNSIGNED_CHAR, 0,  args->tag, *(args->comm));
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0,  args->tag + 1, *(args->comm));

  free(arg);
}

void gpuHostToDeviceTaskAsync(void* arg) {
  gpu_host_to_device_args_t *args = (gpu_host_to_device_args_t*) arg;

  cudaEventSynchronize(args->event);
  cudaError_t e = cudaGetLastError();

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0,  args->tag + 1, *(args->comm));

  freeGpuMemcpyBuffer(&args->buffer);
  free(arg);
}

void gpuDeviceToHostTaskAsync(void* arg) {
  gpu_device_to_host_args_t *args = (gpu_device_to_host_args_t*) arg;

  cudaEventSynchronize(args->event);
  cudaError_t e = cudaGetLastError();
  
  if (e == cudaSuccess){
    MPI_Send((unsigned char*)(args->buffer.buffer), args->count , MPI_UNSIGNED_CHAR, 0,  args->tag, *(args->comm));
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0,  args->tag + 1, *(args->comm));

  freeGpuMemcpyBuffer(&args->buffer);
  free(arg);
}

void logGpuMemcpyError(cudaError_t e, int tag) {
  log_message(LOG_ERROR, "logGpuMemcpyError: Error allocating host memory");
  // Just send the error response
  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, tag + 1, __cudampi__communicators[omp_get_thread_num()]);
}

void cpuLaunchKernelTask(void* arg) {
  launch_kernel_args_t *args = (launch_kernel_args_t*) arg; 
  launchcpukernel(args->kernelArg, args->batchSize, __cudampi__localFreeThreadCount - 1, args->id);
  free(arg);
}

void allocateCpuTaskInStream(void (*task_func)(void *), void *arg, unsigned long stream)
{
  // This function takes a function pointer and argument and adds task to execute it to the list
  // Execution can be done remotely by another thread signaled with task_available_lock
  if (stream >= ALL_CPU_STREAMS) {
    log_message(LOG_ERROR, "Trying to allocate task in invalid stream");
    return;
  }

  task_queue_entry_t *new_dep = (task_queue_entry_t *)malloc(sizeof(task_queue_entry_t));
  if (new_dep == NULL) {
    log_message(LOG_ERROR, "Error allocating memory for dependency node.\n");
    return;
  }
  new_dep->dep_var = 0;  // Initialize dep_var
  new_dep->id = debugTaskCounter++; // Assign ID and increment the counter
  new_dep->task_func = task_func;
  new_dep->arg = arg;
  new_dep->scheduled = 0;   // Task is not scheduled yet

  // Lock to safely update the dependency chain
  omp_set_lock(&queue_locks[stream]);

  // Try setting lock if it is not set (no tasks in the queue);
  omp_test_lock(&synchronize_locks[stream]);

  TAILQ_INSERT_TAIL(&task_queues[stream], new_dep, entries);

  // Signal the task_launcher that a task is available
  // Unlock task_available_lock to allow task_launcher to proceed
  log_message(LOG_DEBUG, "Unsetting task lock\n");
  omp_unset_lock(&task_available_locks[stream]);

  omp_unset_lock(&queue_locks[stream]);

  log_message(LOG_DEBUG, "Created CPU task dependency node. Task ID = %d\n", new_dep->id);
}

void scheduleCpuTask(task_queue_entry_t* current_task_queue_entry, unsigned long stream)
{
  // This function exists, so that independent copy of current_task_queue_entry would be created.
  // Caller of this function moves head pointer of the list by one, but does not remove the node,
  // so there can be a situation that list will be empty (head == tail == NULL),
  // but there will be multiple existing nodes waiting for execution.
  // Nodes can be freed only after the next node after it finished execution,
  // so actual memory management (freeing not used nodes, setting pointer to NULL) is done
  // inside tasks started in this function, meaning each task frees it's previous node.

  task_queue_entry_t* prev_node;

  // IMPORTANT: The assumption here is that queue lock is already set when this is called 
  prev_node = TAILQ_PREV(current_task_queue_entry, task_queue_head, entries);
  current_task_queue_entry->scheduled = 1;
  scheduledTasksInStream[stream] += 1;

  // GPU communication tasks should have higher priority
  // because GPU is faster meaning there will be more GPU-related communication
  // which increases importance of these request being services fast 
  int priority = 0;
  if (stream > CPU_STREAMS_SUPPORTED) {
    priority = 100;
  }

  if (prev_node == NULL)
  {
    // First task has no 'in' dependency
    #pragma omp task untied depend(out: current_task_queue_entry->dep_var) priority(priority)
    {
      log_message(LOG_DEBUG, "FIRST Launching CPU task. Task ID = %d\n", current_task_queue_entry->id);

      // Execute the task function
      current_task_queue_entry->task_func(current_task_queue_entry->arg);

      omp_set_lock(&queue_locks[stream]);

      // Free the current dependency node if there are no other tasks depending on it
      if (TAILQ_NEXT(current_task_queue_entry, entries) == NULL){
        TAILQ_REMOVE(&task_queues[stream], current_task_queue_entry, entries);
        
        // No next task meaning queue is empty
        // Set task_available lock here, so that there wouldn't be a scenario
        // that between now and new lock set there would be a task added
        log_message(LOG_DEBUG, "Unsetting task lock inisde launcher\n");
        
        omp_unset_lock(&synchronize_locks[stream]);
        omp_test_lock(&task_available_locks[stream]); 
      }
      scheduledTasksInStream[stream] -= 1;
      omp_unset_lock(&queue_locks[stream]);

      log_message(LOG_DEBUG, "Finished CPU task execution. Task ID = %d\n", current_task_queue_entry->id);
    }
  } else {
    // Subsequent tasks depend on the previous task
    #pragma omp task untied depend(in: prev_node->dep_var) depend(out: current_task_queue_entry->dep_var) priority(priority)
    {
      log_message(LOG_DEBUG, "SECOND Launching CPU task. Task ID = %d\n", current_task_queue_entry->id);

      // Execute the task function
      current_task_queue_entry->task_func(current_task_queue_entry->arg);

      omp_set_lock(&queue_locks[stream]);

      //Free the previous dependency node if it hasn't been freed
      TAILQ_REMOVE(&task_queues[stream], prev_node, entries);

      // Free the current dependency node if there are no other tasks depending on it
      if (TAILQ_NEXT(current_task_queue_entry, entries) == NULL){
        TAILQ_REMOVE(&task_queues[stream], current_task_queue_entry, entries);

        // No next task meaning queue is empty
        // Set task_available lock here, so that there wouldn't be a scenario
        // that between now and new lock set there would be a task added
        log_message(LOG_DEBUG, "Unsetting task lock inisde launcher\n");
        
        omp_unset_lock(&synchronize_locks[stream]);
        omp_test_lock(&task_available_locks[stream]); 
      }

      scheduledTasksInStream[stream] -= 1;
      omp_unset_lock(&queue_locks[stream]);

      log_message(LOG_DEBUG, "Finished CPU task execution. Task ID = %d\n", current_task_queue_entry->id);
    }
  }
}

void cpuTaskLauncher(unsigned long stream)
{
  // The assumption here is that there is only one thread executing this code.
  // This loop processes the entire FIFO by scheduling taks and exits.

  task_queue_entry_t *current_task_queue_entry;

  omp_set_lock(&queue_locks[stream]);

  // Iterate over the queue and schedule unscheduled tasks
  TAILQ_FOREACH(current_task_queue_entry, &task_queues[stream], entries) {
      if (current_task_queue_entry->scheduled == 0) {
          scheduleCpuTask(current_task_queue_entry, stream);
      }
  }
  omp_unset_lock(&queue_locks[stream]);
}

int main(int argc, char **argv) {

  // basically this is a slave process that waits for requests and redirects
  // those to local GPU(s)
  for (int i = 0; i < ALL_CPU_STREAMS; i++)
  {
    
    scheduledTasksInStream[i] = 0;
    omp_init_lock(&queue_locks[i]);
    omp_init_lock(&synchronize_locks[i]);
    omp_init_lock(&task_available_locks[i]);
    // Set the lock since there are no tasks in queue
    omp_set_lock(&task_available_locks[i]);
    TAILQ_INIT(&task_queues[i]);
  }


  for (int i = 0; i < MAX_THREADS; i++) {
    omp_init_lock(&cpuEnergyLock[i]);
  }

  int mtsprovided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mtsprovided);

  if (mtsprovided != MPI_THREAD_MULTIPLE) {
    log_message(LOG_ERROR, "\nNo support for MPI_THREAD_MULTIPLE mode.\n");
    exit(-1);
  }

  // enable nested omp parallels
  omp_set_nested(1);

  // fetch information about the rank and number of processes

  MPI_Comm_size(MPI_COMM_WORLD, &__cudampi__MPIproccount);
  MPI_Comm_rank(MPI_COMM_WORLD, &__cudampi__myrank);

  // now check the number of GPUs available and synchronize with process 0 that wants this info

  if (cudaSuccess != cudaGetDeviceCount(&__cudampi__localGpuDeviceCount)) {
    log_message(LOG_ERROR, "Error invoking cudaGetDeviceCount()");
    exit(-1);
  }

  __cudampi__GPUcountspernode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__GPUcountspernode) {
    log_message(LOG_ERROR, "\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  __cudampi__freeThreadsPerNode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__freeThreadsPerNode) {
    log_message(LOG_ERROR, "\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  MPI_Bcast(&__cudampi__cpu_enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&__cudampi__cpu_power_scaling, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (__cudampi__cpu_enabled){
      assert(__cudampi__cpu_power_scaling > 0.0 && __cudampi__cpu_power_scaling <= 1.0);
  
      if (cudaSuccess != __cudampi__getCpuFreeThreads(&__cudampi__localFreeThreadCount)) {
      log_message(LOG_ERROR, "Error invoking __cudampi__getCpuFreeThreads()");
      exit(-1);
    }
  }
  else {
    __cudampi__localFreeThreadCount = 0;
  }

  MPI_Allgather(&__cudampi__localGpuDeviceCount, 1, MPI_INT, __cudampi__GPUcountspernode, 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Allgather(&__cudampi__localFreeThreadCount, 1, MPI_INT, __cudampi__freeThreadsPerNode, 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Bcast(&__cudampi__totaldevicecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  __cudampi_targetMPIrankfordevice = (int *)malloc(__cudampi__totaldevicecount * sizeof(int));
  if (!__cudampi_targetMPIrankfordevice) {
    log_message(LOG_ERROR, "\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  MPI_Bcast(__cudampi_targetMPIrankfordevice, __cudampi__totaldevicecount, MPI_INT, 0, MPI_COMM_WORLD);

  // create communicators
  // in the case of the slave we need to go by every GPU and for each GPU there will be a separate GPU shared with the master -- process 0

  MPI_Comm *__cudampi__communicators = (MPI_Comm *)malloc(sizeof(MPI_Comm) * __cudampi__totaldevicecount);
  if (!__cudampi__communicators) {
    log_message(LOG_ERROR, "\nNot enough memory for communicators");
    exit(-1); // we could exit in a nicer way! TBD
  }

  int commcounter = 0;

  for (int i = __cudampi__GPUcountspernode[0]; i < __cudampi__totaldevicecount; i++) {

    int ranks[2] = {0, __cudampi_targetMPIrankfordevice[i]}; // group and communicator between process 0 and the process of the target GPU/device

    MPI_Group groupall;
    MPI_Comm_group(MPI_COMM_WORLD, &groupall);

    // Keep only process 0 and the process handling the given GPU
    MPI_Group tempgroup;
    MPI_Group_incl(groupall, 2, ranks, &tempgroup);

    MPI_Comm tempcomm;
    if (__cudampi_targetMPIrankfordevice[i] != __cudampi__myrank) {
      MPI_Comm_create(MPI_COMM_WORLD, tempgroup, &tempcomm);
    } else {
      MPI_Comm_create(MPI_COMM_WORLD, tempgroup, &(__cudampi__communicators[commcounter++]));
    }
  }

  // here we need to spawn threads -- each responsible for handling one local GPU / CPU
  // spawn one thread for CPU processing if there are free cores, 0 otherwise
  int numberOfCpuThreads = __cudampi__localFreeThreadCount > 0;
  int deviceThreads = numberOfCpuThreads + __cudampi__localGpuDeviceCount; 
  int numberOfThreads = deviceThreads;
  
  // Add one thread for executing CPU operations asynchronously
  if (CPU_STREAMS_SUPPORTED < 1 || CPU_STREAMS_SUPPORTED > 2) {
    log_message(LOG_ERROR, "It is only possible to launch 1 or 2 CPU streams. Currently attempted: %d", CPU_STREAMS_SUPPORTED);
  }
  numberOfThreads += ALL_CPU_STREAMS - (MAX_GPU_PER_NODE - __cudampi__localGpuDeviceCount);

  // If CPU disabled, terminate task handling threads
  if(numberOfCpuThreads == 0) {
    for (int i = 0; i < CPU_STREAMS_SUPPORTED; i++) {
      terminated[i] = 1;
    }
  }

  // initialize nvml
  nvmlReturn_t nvmlResult = nvmlInit();
  if (nvmlResult != NVML_SUCCESS) {
      log_message(LOG_ERROR, "nvmlInit failed: %s\n", nvmlErrorString(nvmlResult));
      return -1;
  }
  
  assert(numberOfThreads < MAX_THREADS);

  #pragma omp parallel num_threads(numberOfThreads)
  {

    if (omp_get_thread_num() < __cudampi__localGpuDeviceCount) {
      // GPU handling thread
      MPI_Status status;
      globalGpuMemcpyBuffer gpuMemcpyBuffer;
      initializeGlobalGpuMemcpyBuffer(&gpuMemcpyBuffer, INITIAL_GPU_BUFFER_SIZE);

      do {

        MPI_Probe(0, MPI_ANY_TAG, __cudampi__communicators[omp_get_thread_num()], &status);

        if (status.MPI_TAG == __cudampi__CUDAMPIMALLOCREQ) {
          unsigned long rdata;

          MPI_Recv((unsigned long *)(&rdata), 1, MPI_UNSIGNED_LONG, 0, __cudampi__CUDAMPIMALLOCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          // allocate memory on the current GPU
          void *devPtr;

          cudaError_t e = cudaMalloc(&devPtr, (size_t)rdata);

          int ssize = sizeof(void *) + sizeof(cudaError_t);
          // send confirmation with the actual pointer
          unsigned char sdata[ssize];

          *((void **)sdata) = devPtr;
          *((cudaError_t *)(sdata + sizeof(void *))) = e;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIMALLOCRESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPIFREEREQ) {
          int rsize = sizeof(void *);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIFREEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);

          cudaError_t e = cudaFree(devPtr);

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIFREERESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ) {
          int measurepower;
          cudaError_t error = cudaErrorUnknown;

          MPI_Recv(&measurepower, 1, MPI_INT, 0, __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          // perform power measurement and attach it to the response

          size_t ssize = sizeof(cudaError_t) + sizeof(float) + sizeof(float);
          unsigned char sdata[ssize];
          float gpuPowerMeasured = -1.0;
          float cpuEnergyMeasured = -1.0;

          int device;
          cudaGetDevice(&device);

          cudaError_t e = cudaDeviceSynchronize();

          // Synchronize async memcpy tasks
          omp_set_lock(&synchronize_locks[CPU_STREAM_FOR_GPU_RESPONSES]);
          // Free the lock back
          omp_unset_lock(&synchronize_locks[CPU_STREAM_FOR_GPU_RESPONSES]);

          *((cudaError_t *)sdata) = e;
            
          if (measurepower) {
            error = getCpuEnergyUsed(&cpuLastEnergyMeasured[omp_get_thread_num()], &cpuEnergyMeasured);
            if (error != cudaSuccess) {
              cpuEnergyMeasured = -1.0;
            }
            else {
              cpuEnergyMeasured *= ((1 - __cudampi__cpu_power_scaling) / __cudampi__localGpuDeviceCount);
            }
            gpuPowerMeasured = getGPUpower(device);
          }

          *((float *)(sdata + sizeof(cudaError_t))) = gpuPowerMeasured;
          *((float *)(sdata + sizeof(cudaError_t) + sizeof(float))) = cpuEnergyMeasured;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICESYNCHRONIZERESP, __cudampi__communicators[omp_get_thread_num()]);
          updateGlobalGpuMemcpyBuffer(&gpuMemcpyBuffer);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPISETDEVICEREQ) {
          int device;

          MPI_Recv(&device, 1, MPI_INT, 0, __cudampi__CUDAMPISETDEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          cudaError_t e = cudaSetDevice(device);

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISETDEVICERESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPIHOSTTODEVICEREQ) {
          // in this case in the message there is a serialized pointer and data so we need to find out the size first

          int rsize;
          MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

          unsigned char *rdata = malloc(rsize);

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);

          // now send the data to the GPU
          cudaError_t e = cudaMemcpy(devPtr, rdata + sizeof(void *), rsize - sizeof(void *), cudaMemcpyHostToDevice);

          if (cudaSuccess != cudaGetLastError()) {
            log_message(LOG_ERROR, "\nError xxx host to dev");
          }

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICERESP, __cudampi__communicators[omp_get_thread_num()]);

          free(rdata);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ) {
          // in this case in the message there is a serialized pointer and data so we need to find out the size first

          int rsize;
          MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

          gpu_host_to_device_args_t *args = malloc(sizeof(gpu_host_to_device_args_t));
          if (args == NULL) {
            log_message(LOG_ERROR, "Error allocating memory");
            continue;
          }
          args->buffer = allocateGpuMemcpyBuffer(&gpuMemcpyBuffer, rsize);
          unsigned char *rdata = args->buffer.buffer;

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);
          cudaStream_t stream = *((cudaStream_t *)(rdata + sizeof(void *)));
          int tag = *((int *)(rdata + sizeof(void *) + sizeof(cudaStream_t)));

          // now send the data to the GPU
          cudaEvent_t event;
          cudaEventCreate(&event);

          cudaError_t e = cudaMemcpyAsync(devPtr, rdata + sizeof(void *) + sizeof(cudaStream_t) + sizeof(int), rsize - sizeof(void *) - sizeof(cudaStream_t) - sizeof(int), cudaMemcpyHostToDevice, stream);
          cudaEventRecord(event, stream);

          if (e != cudaSuccess) {
            logGpuMemcpyError(e, tag);
            freeGpuMemcpyBuffer(&args->buffer);
            continue;
          }

          // Schedule a task that would wait for the copy to complete and send back the response
          args->tag = tag;
          args->comm = &__cudampi__communicators[omp_get_thread_num()];
          args->event = event;

          allocateCpuTaskInStream(gpuHostToDeviceTaskAsync, args, CPU_STREAM_FOR_GPU_RESPONSES);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPIDEVICETOHOSTREQ) {
          // in this case in the message there is a serialized pointer and size of data to fetch

          int rsize = sizeof(void *) + sizeof(unsigned long);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);
          unsigned long count = *((unsigned long *)(rdata + sizeof(void *)));

          size_t ssize = sizeof(cudaError_t) + count;
          unsigned char *sdata = malloc(ssize);

          // now send the data to the GPU
          cudaError_t e = cudaMemcpy(sdata + sizeof(cudaError_t), devPtr, count, cudaMemcpyDeviceToHost);

          if (cudaSuccess != cudaGetLastError()) {
            log_message(LOG_ERROR, "\nError yyy dev to host");
          }

          *((cudaError_t *)sdata) = e;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTRESP, __cudampi__communicators[omp_get_thread_num()]);

          free(sdata);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ) {
          // in this case in the message there is a serialized pointer and size of data to fetch

          int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t) + sizeof(int);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);
          unsigned long count = *((unsigned long *)(rdata + sizeof(void *)));
          cudaStream_t stream = *((cudaStream_t *)(rdata + sizeof(void *) + sizeof(unsigned long)));
          int tag = *((int *)(rdata + sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t)));

          gpu_device_to_host_args_t *args = malloc(sizeof(gpu_device_to_host_args_t));
          if (args == NULL) {
            log_message(LOG_ERROR, "Error allocating memory");
            continue;
          }

          args->buffer = allocateGpuMemcpyBuffer(&gpuMemcpyBuffer, count);
          unsigned char *sdata = args->buffer.buffer;

          cudaEvent_t event;
          cudaEventCreate(&event);

          // now send the data to the GPU
          cudaError_t e = cudaMemcpyAsync(args->buffer.buffer, devPtr, count, cudaMemcpyDeviceToHost, stream);
          cudaEventRecord(event, stream);

          if (e != cudaSuccess) {
            logGpuMemcpyError(e, tag);
            freeGpuMemcpyBuffer(&args->buffer);
            continue;
          }

          // Schedule a task that would wait for the copy to complete and send back the response
          args->tag = tag;
          args->comm = &__cudampi__communicators[omp_get_thread_num()];
          args->event = event;
          args->count = count;

          allocateCpuTaskInStream(gpuDeviceToHostTaskAsync, args, CPU_STREAM_FOR_GPU_RESPONSES);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPILAUNCHCUDAKERNELREQ) {
          // in this case in the message there is a serialized pointer
          
          initializeCpuEnergyMeasurement(isInitialCpuEnergyMeasured, cpuEnergyLock, cpuLastEnergyMeasured);

          int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long long);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHCUDAKERNELREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);
          unsigned long batchSize = *((unsigned long *)(rdata + sizeof(void *)));
          unsigned long long id = *((unsigned long long *)(rdata + sizeof(void *) + sizeof(unsigned long)));

          launchkernel(devPtr, batchSize, id);

          size_t ssize = sizeof(cudaError_t);
          unsigned char sdata[ssize];
          *((cudaError_t *)sdata) = cudaSuccess;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHCUDAKERNELRESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ) {
          // in this case in the message there is a serialized pointer

          initializeCpuEnergyMeasurement(isInitialCpuEnergyMeasured, cpuEnergyLock, cpuLastEnergyMeasured);

          int rsize = sizeof(void *) + sizeof(cudaStream_t) + sizeof(unsigned long) + sizeof(unsigned long long);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          void *devPtr = *((void **)rdata);
          unsigned long batchSize = *((unsigned long *)(rdata + sizeof(void *)));
          cudaStream_t stream = *((cudaStream_t *)(rdata + sizeof(void *) + sizeof(unsigned long)));
          unsigned long long id = *((unsigned long long *)(rdata + sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t)));

          launchkernelinstream(devPtr, batchSize, stream, id);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPISTREAMCREATEREQ) {
          MPI_Recv(NULL, 0, MPI_UNSIGNED_LONG, 0, __cudampi__CUDAMPISTREAMCREATEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          // create a new stream on the current GPU

          cudaStream_t pStream;

          cudaError_t e = cudaStreamCreate(&pStream);

          int ssize = sizeof(cudaStream_t) + sizeof(cudaError_t);
          // send confirmation with the actual pointer
          unsigned char sdata[ssize];

          *((cudaStream_t *)sdata) = pStream;
          *((cudaError_t *)(sdata + sizeof(cudaStream_t))) = e;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISTREAMCREATERESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPISTREAMDESTROYREQ) {
          int rsize = sizeof(cudaStream_t);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISTREAMDESTROYREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          cudaStream_t stream = *((cudaStream_t *)rdata);

          cudaError_t e = cudaStreamDestroy(stream);

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISTREAMDESTROYRESP, __cudampi__communicators[omp_get_thread_num()]);
        }
      } while (status.MPI_TAG != __cudampi__CUDAMPIFINALIZE);

      terminated[CPU_STREAM_FOR_GPU_RESPONSES] = 1;
      omp_unset_lock(&task_available_locks[CPU_STREAM_FOR_GPU_RESPONSES]);
      freeGlobalGpuMemcpyBuffer(&gpuMemcpyBuffer);
    }
    else if (omp_get_thread_num() < deviceThreads) {
      // CPU handling thread
      MPI_Status status;
      do {
        MPI_Probe(0, MPI_ANY_TAG, __cudampi__communicators[omp_get_thread_num()], &status);

        if (status.MPI_TAG == __cudampi__CPUMALLOCREQ) {
          unsigned long rdata;

          MPI_Recv((unsigned long *)(&rdata), 1, MPI_UNSIGNED_LONG, 0, __cudampi__CPUMALLOCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          cpuSynchronize();
          log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUMALLOCREQ\n");

          void *devPtr = malloc((size_t)rdata);

          int ssize = sizeof(void *) + sizeof(cudaError_t);
          // send confirmation with the actual pointer
          unsigned char sdata[ssize];

          *((void **)sdata) = devPtr;

          // return cudaSuccess if memory is not NULL, cudaErrorMemoryAllocation otherwise
          cudaError_t e = (devPtr == NULL) ? cudaErrorMemoryAllocation : cudaSuccess;
          *((cudaError_t *)(sdata + sizeof(void *))) = e;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUMALLOCRESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CPUFREEREQ) {
          int rsize = sizeof(void *);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUFREEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          cpuSynchronize();
          log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUFREEREQ\n");

          void *devPtr = *((void **)rdata);
          cudaError_t e = cudaErrorInvalidDevicePointer;

          // maybe we should somehow handle invalid devPtr ?
          if (devPtr != NULL) {
            free(devPtr);
            e = cudaSuccess;
          }

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUFREERESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CUDAMPICPUDEVICESYNCHRONIZEREQ) {
          int measurepower;
          cudaError_t error = cudaErrorUnknown;

          MPI_Recv(&measurepower, 1, MPI_INT, 0, __cudampi__CUDAMPICPUDEVICESYNCHRONIZEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          // perform power measurement and attach it to the response

          size_t ssize = sizeof(cudaError_t) + sizeof(float);
          unsigned char sdata[ssize];
          float energyMeasured;

          cpuSynchronize();

          if (measurepower) {
            error = getCpuEnergyUsed(&cpuLastEnergyMeasured[omp_get_thread_num()], &energyMeasured);
          }

          if (error != cudaSuccess) {
            energyMeasured = -1.0;
          }
          else {
            energyMeasured *= __cudampi__cpu_power_scaling;
          }

          *((cudaError_t *)sdata) = error;
          *((float *)(sdata + sizeof(cudaError_t))) = energyMeasured;

          log_message(LOG_DEBUG, "Synchronized CPU device\n");
          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPICPUDEVICESYNCHRONIZERESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CPUHOSTTODEVICEREQ) {
          int rsize;
          MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

          unsigned char *rdata = malloc(rsize);
          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          cpuSynchronize();
          log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUHOSTTODEVICEREQ\n");

          void *devPtr = *((void **)rdata);
          void *srcPtr = rdata + sizeof(void *);
          size_t count = rsize - sizeof(void *);

          cudaError_t e = cudaErrorInvalidValue;

          if (devPtr != NULL && srcPtr != NULL && count > 0) {
            memcpy(devPtr, srcPtr, rsize - sizeof(void *));
            e = cudaSuccess;
          }

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICERESP, __cudampi__communicators[omp_get_thread_num()]);

          free(rdata);
        }

        if (status.MPI_TAG == __cudampi__CPUDEVICETOHOSTREQ) {
          int rsize = sizeof(void *) + sizeof(unsigned long);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          cpuSynchronize();
          log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUDEVICETOHOSTREQ\n");

          void *devPtr = *((void **)rdata);
          unsigned long count = *((unsigned long *)(rdata + sizeof(void *)));

          cudaError_t e = cudaErrorInvalidValue;

          size_t ssize = sizeof(cudaError_t) + count;
          unsigned char *sdata = malloc(ssize);

          if (devPtr != NULL && count > 0) {
            memcpy(sdata + sizeof(cudaError_t), devPtr, count);
            e = cudaSuccess;
          }

          *((cudaError_t *)sdata) = e;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTRESP, __cudampi__communicators[omp_get_thread_num()]);

          free(sdata);
        }

        if (status.MPI_TAG == __cudampi__CPUHOSTTODEVICEREQASYNC) {
          int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long) + sizeof(int);
          // Receive a request with number of bytes that will be sent and a pointer
          unsigned char rdata[rsize];
          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICEREQASYNC, __cudampi__communicators[omp_get_thread_num()], &status);

          // Launch a task that will receive the data
          cpu_host_to_device_args_t *args = malloc(sizeof(cpu_host_to_device_args_t));
          args->devPtr = *((void **)rdata);
          args->count = *((unsigned long *)(rdata + sizeof(void *)));
          unsigned long stream = *((unsigned long *)(rdata + sizeof(void *) + sizeof(unsigned long)));
          args->tag = *((int *)(rdata + sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long)));
          args->comm = &__cudampi__communicators[omp_get_thread_num()];

          log_message(LOG_DEBUG, "Allocating CPU task for __cudampi__CPUHOSTTODEVICEREQASYNC in stream %d\n", stream);
          allocateCpuTaskInStream(cpuHostToDeviceTaskAsync, args, stream);
        }

        if (status.MPI_TAG == __cudampi__CPUDEVICETOHOSTREQASYNC) {
          int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long) + sizeof(int);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTREQASYNC, __cudampi__communicators[omp_get_thread_num()], &status);

          cpu_device_to_host_args_t *args = malloc(sizeof(cpu_device_to_host_args_t));
          args->devPtr = *((void **)rdata);
          args->count = *((unsigned long *)(rdata + sizeof(void *)));
          unsigned long stream = *((unsigned long *)(rdata + sizeof(void *) + sizeof(unsigned long)));
          args->tag = *((int *)(rdata + sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long)));
          args->comm = &__cudampi__communicators[omp_get_thread_num()];

          log_message(LOG_DEBUG, "Allocating CPU task for __cudampi__CPUDEVICETOHOSTREQASYNC in stream %d\n", stream);
          allocateCpuTaskInStream(cpuDeviceToHostTaskAsync, args, stream);
        }

        if (status.MPI_TAG == __cudampi__CPULAUNCHKERNELREQ) {
          initializeCpuEnergyMeasurement(isInitialCpuEnergyMeasured, cpuEnergyLock, cpuLastEnergyMeasured);

          int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long) + sizeof(unsigned long long);

          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPULAUNCHKERNELREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          
          launch_kernel_args_t *args = malloc(sizeof(cpu_host_to_device_args_t));
          args->kernelArg = *((void **)rdata);
          args->batchSize = *((unsigned long *)(rdata + sizeof(void *)));

          unsigned long stream = *((unsigned long *)(rdata + sizeof(void *) + sizeof(unsigned long)));
  
          args->id = *((unsigned long long *)(rdata + sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long)));

          log_message(LOG_DEBUG, "Allocating CPU task for __cudampi__CPULAUNCHKERNELREQ in stream %d\n", stream);
          allocateCpuTaskInStream(cpuLaunchKernelTask, args, stream);
        }

        if (status.MPI_TAG == __cudampi__CPUSTREAMCREATEREQ) {
          MPI_Recv(NULL, 0, MPI_UNSIGNED_LONG, 0, __cudampi__CPUSTREAMCREATEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          // create a new stream on CPU
          cudaError_t e = cudaErrorInvalidResourceHandle;

          int freeStreamSlot = -1;
          for (int i = 0; i < CPU_STREAMS_SUPPORTED; i++) {
            // Find free stream slot
            if (cpuStreamsValid[i] == 0) {
              cpuStreamsValid[i] = 1;
              freeStreamSlot = i;
              e = cudaSuccess;
              break;
            }
          }

          int ssize = sizeof(unsigned long) + sizeof(cudaError_t);
          // send confirmation with the actual pointer
          unsigned char sdata[ssize];

          *((unsigned long *)sdata) = freeStreamSlot;
          *((cudaError_t *)(sdata + sizeof(cudaStream_t))) = e;

          MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUSTREAMCREATERESP, __cudampi__communicators[omp_get_thread_num()]);
        }

        if (status.MPI_TAG == __cudampi__CPUSTREAMDESTROYREQ) {
          int rsize = sizeof(unsigned long);
          unsigned char rdata[rsize];

          MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUSTREAMDESTROYREQ, __cudampi__communicators[omp_get_thread_num()], &status);

          unsigned long stream = *((unsigned long *)rdata);

          cudaError_t e = cudaErrorInvalidResourceHandle;

          if (stream < CPU_STREAMS_SUPPORTED) {
            cpuStreamsValid[stream] = 0;
            e = cudaSuccess;
          }

          MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUSTREAMDESTROYRESP, __cudampi__communicators[omp_get_thread_num()]);
        }

      } while (status.MPI_TAG != __cudampi__CUDAMPIFINALIZE);

      for (int i = 0; i < CPU_STREAMS_SUPPORTED; i++) {
        terminated[i] = 1;
        omp_unset_lock(&task_available_locks[i]);
      }
    }
    else {
      // Task execution thread
      int localTaskCounter = 0;
      int stream = omp_get_thread_num() - deviceThreads;

      while (!terminated[stream]) {
        // If there's data in task queue, wait for completion
        omp_set_lock(&queue_locks[stream]);
        localTaskCounter = scheduledTasksInStream[stream];
        omp_unset_lock(&queue_locks[stream]);

        log_message(LOG_DEBUG, "Tasks in queue: %d\n", localTaskCounter);

        if (localTaskCounter > 0) {
          log_message(LOG_DEBUG, "Waiting for task execution \n");
          #pragma omp taskwait
        } else if(!terminated[stream]) {

          log_message(LOG_DEBUG, "Setting task lock \n");
          // Wait for new tasks to be allocated
          omp_set_lock(&task_available_locks[stream]);
          // Call a function that schedules all the tasks in the queue
          cpuTaskLauncher(stream);
        }
      }
      log_message(LOG_DEBUG, "Terminated task handling thread!");
      omp_destroy_lock(&task_available_locks[stream]);
    }
  }

  nvmlShutdown();
  MPI_Finalize();
  
  for (int i = 0; i < ALL_CPU_STREAMS; i++)
  {
    omp_destroy_lock(&queue_locks[i]);
    omp_destroy_lock(&synchronize_locks[i]);
    omp_destroy_lock(&task_available_locks[i]);
  }
  
  for (int i = 0; i < MAX_THREADS; i++) {
    omp_destroy_lock(&cpuEnergyLock[i]);
  }
}
