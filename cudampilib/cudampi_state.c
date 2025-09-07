// Definitions of shared CUDAMPI state (master side)
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>

#include "cudampi_state.h"

int *__cudampi__GPUcountspernode;
int *__cudampi__CPUcountspernode;
int *__cudampi__freeThreadsPerNode;

int __cudampi_totaldevicecount = 0;
int __cudampi_totalgpudevicecount = 0;
int __cudampi_totalcpudevicecount = 0;

int __cudampi__localGpuDeviceCount = 0;
int __cudampi__localFreeThreadCount = 0;

int *__cudampi_targetGPUfordevice;
int *__cudampi_targetMPIrankfordevice;

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

int __cudampi__currentdevice[__CUDAMPI_MAX_THREAD_COUNT];
struct timeval __cudampi__timestart[__CUDAMPI_MAX_THREAD_COUNT];
struct timeval __cudampi__timestop[__CUDAMPI_MAX_THREAD_COUNT];
double __cudampi__time_us[__CUDAMPI_MAX_THREAD_COUNT];
int __cudampi__timemeasured[__CUDAMPI_MAX_THREAD_COUNT] = {0};

devicePowerConfig_t __cudampi__devicePowerConfig[__CUDAMPI_MAX_THREAD_COUNT];
omp_lock_t __cudampi__devicelocks[__CUDAMPI_MAX_THREAD_COUNT];
omp_lock_t deviceselectionlock;
int __cudampi__amimanager[__CUDAMPI_MAX_THREAD_COUNT] = {0};

MPI_Comm *__cudampi__communicators;

float __cudampi__globalpowerlimit = -1.0f;

unsigned long __cudampi__batches_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long long __cudampi__data_points_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long __cudampi__last_batches_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long __cudampi__mgr_batches_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long __cudampi__mgr_data_points_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};
unsigned long long __cudampi__last_data_points_sent[__CUDAMPI_MAX_THREAD_COUNT] = {0};

unsigned long __cudampi__default_batch_size;
unsigned long __cudampi__cpu_batch_size;
float __cudampi__cpu_power_scaling;
int __cudampi__cpu_enabled;

