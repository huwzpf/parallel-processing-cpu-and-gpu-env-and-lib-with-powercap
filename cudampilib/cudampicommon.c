/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "cudampicommon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <nvml.h>
#define ENABLE_LOGGING
#define MPI_LOGGING
#include "logger.h"

float computeDevPerformance(double period_us) {
  // period is just the time between two events so compute performance as an inverse

  return 1000000.0 / period_us;
}

float getGPUpower(int gpuid) {
    nvmlReturn_t result;
    unsigned int power_mw;
    float power_watts;
    nvmlDevice_t nvmlDevice;

    result = nvmlDeviceGetHandleByIndex(gpuid, &nvmlDevice);
    if (result != NVML_SUCCESS) {
        log_message(LOG_ERROR, "nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(result));
        return -1;
    }

    result = nvmlDeviceGetPowerUsage(nvmlDevice, &power_mw);
    if (result != NVML_SUCCESS) {
        log_message(LOG_ERROR, "Failed to get power usage: %s\n", nvmlErrorString(result));
        return -1;
    }

    return (float)power_mw / 1000.0;
}

cudaError_t __cudampi__getCpuFreeThreads(int* count)
{
  int gpuCount = 0;
  cudaError_t status = cudaGetDeviceCount(&gpuCount);
  *count = omp_get_max_threads() - (gpuCount * 2);
  return status;
}

 cudaError_t getCpuEnergyUsed(float* lastEnergyMeasured, float* energyUsed) {
  // compute energy used from last energy measurement and update the variable

  FILE *file;
  unsigned long long energy_uj;
  float energy_joules;

  file = fopen("/sys/class/powercap/intel-rapl:0/energy_uj", "r");
  if (file == NULL) {
      log_message(LOG_ERROR, "Failed to open energy_uj file");
      return cudaErrorUnknown ;
  }

  if (fscanf(file, "%llu", &energy_uj) != 1) {
      log_message(LOG_ERROR, "Failed to read energy value");
      fclose(file);
      return cudaErrorUnknown ;
  }

  fclose(file);
  log_message(LOG_DEBUG, "Got energy_uj = %lld", energy_uj);
  energy_joules = (float)energy_uj / 1e6;

  *energyUsed = energy_joules - *lastEnergyMeasured;

  *lastEnergyMeasured = energy_joules;

  return cudaSuccess;
}

void initializeCpuEnergyMeasurement(int* isInitialCpuEnergyMeasured, omp_lock_t* cpuEnergyLock, float* cpuLastEnergyMeasured) {
  // Each thread executes this function before kernel launch to make sure that cpu energy was initialized
  if (!isInitialCpuEnergyMeasured[omp_get_thread_num()]) {
    // Initialize CPU energy value
    omp_set_lock(&cpuEnergyLock[omp_get_thread_num()]);
    if (!isInitialCpuEnergyMeasured[omp_get_thread_num()]) {
      // This variable is unused since we just need to initialize cpuLastEnergyMeasured and don't care about actual value
      float cpuEnergyMeasured;
      isInitialCpuEnergyMeasured[omp_get_thread_num()] = 1;
      getCpuEnergyUsed(&cpuLastEnergyMeasured[omp_get_thread_num()], &cpuEnergyMeasured);
    }
    omp_unset_lock(&cpuEnergyLock[omp_get_thread_num()]);
  }
}
