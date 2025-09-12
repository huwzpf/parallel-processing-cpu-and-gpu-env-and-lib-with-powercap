/* Centralised shared state for CUDAMPI master side */
#ifndef CUDAMPI_STATE_H
#define CUDAMPI_STATE_H

#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

#include "cudampi.h"
#include "cudampicommon.h"

// Thread-local access helpers to shared state
#define __cudampi__currentDevice  __cudampi__currentdevice[omp_get_thread_num()]
#define __cudampi__currentCommunicator  __cudampi__communicators[__cudampi__currentDevice]
#define __cudampi_isLocalGpu (__cudampi__currentDevice < __cudampi__GPUcountspernode[0])

// Node/device topology
extern int *__cudampi__GPUcountspernode;
extern int *__cudampi__CPUcountspernode;
extern int *__cudampi__freeThreadsPerNode;
extern int __cudampi_totaldevicecount;
extern int __cudampi_totalgpudevicecount;
extern int __cudampi_totalcpudevicecount;
extern int __cudampi__localGpuDeviceCount;
extern int __cudampi__localFreeThreadCount;
extern int *__cudampi_targetGPUfordevice;
extern int *__cudampi_targetMPIrankfordevice;

// MPI state
extern int __cudampi__MPIinitialized;
extern int __cudampi__MPIproccount;
extern int __cudampi__myrank;
extern MPI_Comm *__cudampi__communicators;

// Device selection and metrics
extern int __cudampi__currentdevice[__CUDAMPI_MAX_THREAD_COUNT];
extern struct timeval __cudampi__timestart[__CUDAMPI_MAX_THREAD_COUNT];
extern struct timeval __cudampi__timestop[__CUDAMPI_MAX_THREAD_COUNT];
extern double __cudampi__time_us[__CUDAMPI_MAX_THREAD_COUNT];
extern int __cudampi__timemeasured[__CUDAMPI_MAX_THREAD_COUNT];
extern devicePowerConfig_t __cudampi__devicePowerConfig[__CUDAMPI_MAX_THREAD_COUNT];
extern omp_lock_t __cudampi__devicelocks[__CUDAMPI_MAX_THREAD_COUNT];
extern omp_lock_t deviceselectionlock;
extern int __cudampi__amimanager[__CUDAMPI_MAX_THREAD_COUNT];

// Power capping / accounting
extern float __cudampi__globalpowerlimit;
extern unsigned long __cudampi__batches_sent[__CUDAMPI_MAX_THREAD_COUNT];
extern unsigned long long __cudampi__data_points_sent[__CUDAMPI_MAX_THREAD_COUNT];
extern unsigned long __cudampi__last_batches_sent[__CUDAMPI_MAX_THREAD_COUNT];
extern unsigned long __cudampi__mgr_batches_sent[__CUDAMPI_MAX_THREAD_COUNT];
extern unsigned long __cudampi__mgr_data_points_sent[__CUDAMPI_MAX_THREAD_COUNT];
extern unsigned long long __cudampi__last_data_points_sent[__CUDAMPI_MAX_THREAD_COUNT];

// Batch size / CPU scaling
extern unsigned long __cudampi__default_batch_size;
extern unsigned long __cudampi__cpu_batch_size;
extern float __cudampi__cpu_power_scaling;
extern int __cudampi__cpu_enabled;

// First-iteration dynamic scaling stats
extern double __cudampi__firstIterTotalGpuBatchTimeSeconds;
extern double __cudampi__firstIterTotalCpuBatchTimeSeconds;
extern unsigned long __cudampi__firstIterTotalGpuBatches;
extern unsigned long __cudampi__firstIterTotalCpuBatches;
extern int __cudampi__firstIterNumberGpuDevicesMeasured;
extern int __cudampi__firstIterNumberCpuDevicesMeasured;
extern int __cudampi__firstIterDeviceMeasurementStarted[__CUDAMPI_MAX_THREAD_COUNT];
extern int __cudampi__firstIterMeasuredForDevice[__CUDAMPI_MAX_THREAD_COUNT];
extern int __cudampi__cpuBatchSizeScalingDone;
extern int __cudampi__dyanmicCpuBatchSizeScalingEnabled;

// Energy
extern double __cudampi__totalEnergyUsed;

// Optimizer sync period and app start timestamp
extern double __cudampi__optimizerSyncPeriod; // seconds between optimizer syncs
extern int __cudampi__appStartTimestampSet;   // 1 once set globally
extern struct timeval __cudampi__appStartTime; // time of first batch record

// Optimisation completion snapshot (set when edp_optimization_steps reached)
extern int __cudampi__optimizationFinished;                 // 1 once optimisation step limit reached
extern double __cudampi__optimizationFinishedTime;          // seconds since app start at finish
extern double __cudampi__optimizationFinishedEnergy;        // total energy used at finish (J)
extern unsigned long long __cudampi__optimizationFinishedDataPoints; // total data points processed at finish

#endif // CUDAMPI_STATE_H
