/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "cudampicommon.h"
#include <sys/time.h>
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

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#define SOCKET_PATH "/tmp/gpu_power_tool.sock"

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
        log_message(LOG_ERROR, "nvmlDeviceGetHandleByIndex failed: %s", nvmlErrorString(result));
        return -1;
    }

    result = nvmlDeviceGetPowerUsage(nvmlDevice, &power_mw);
    if (result != NVML_SUCCESS) {
        log_message(LOG_ERROR, "Failed to get power usage: %s", nvmlErrorString(result));
        return -1;
    }

    return (float)power_mw / 1000.0;
}

powercapRange_t __cudampi__getCpuPowerCapRange()
{
  powercapRange_t range;
  FILE *file;
  unsigned long long min, max;
  unsigned long long powerCap_uw;
  unsigned long long timeWindow_us;

  range.min = 0; // For now assume that the minimum power cap is 0 as min_power_uw is not available

  file = fopen("/sys/class/powercap/intel-rapl:0/constraint_0_max_power_uw", "r");
  if (file == NULL) {
      log_message(LOG_ERROR, "Failed to open max_power_uw file");
      range.max = -1;
      range.defaultPowerCap = -1;
      return range;
  }

  if (fscanf(file, "%llu", &max) != 1) {
      log_message(LOG_ERROR, "Failed to read max_power_uw value");
      fclose(file);
      range.max = -1;
      range.defaultPowerCap = -1;
      return range;
  }

  fclose(file);

  range.max = (float)max / 1e6;
  file = fopen("/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw", "r");
  if (file == NULL) {
    log_message(LOG_ERROR, "Failed to open constraint_0_power_limit_uw file for reading");
    range.defaultPowerCap = -1;
    return range;
  }

  if (fscanf(file, "%llu", &powerCap_uw) != 1) {
    log_message(LOG_ERROR, "Failed to read power cap value");
    fclose(file);
    range.defaultPowerCap = -1;
    return range;
  }

  fclose(file);

  // range.defaultPowerCap = (float)powerCap_uw / 1e6;
  range.defaultPowerCap = range.max;
  file = fopen("/sys/class/powercap/intel-rapl:0/constraint_0_time_window_us", "r");
  if (file == NULL) {
    log_message(LOG_ERROR, "Failed to open constraint_0_time_window_us file for reading");
    range.timeWindowUs = -1;
    return range;
  }

  if (fscanf(file, "%llu", &timeWindow_us) != 1) {
    log_message(LOG_ERROR, "Failed to read time window value");
    fclose(file);
    range.timeWindowUs = -1;
    return range;
  }

  fclose(file);

  range.timeWindowUs = timeWindow_us;

  return range;
}

powercapRange_t __cudampi__getGpuPowerCapRange(int gpuid)
{
  powercapRange_t range;
  nvmlReturn_t result;
  unsigned int min, max, defaultPower;
  nvmlDevice_t nvmlDevice;

  result = nvmlDeviceGetHandleByIndex(gpuid, &nvmlDevice);
  if (result != NVML_SUCCESS) {
      log_message(LOG_ERROR, "nvmlDeviceGetHandleByIndex failed: %s", nvmlErrorString(result));
      range.min = -1;
      range.max = -1;
      range.defaultPowerCap = -1;
      return range;
  }

  result = nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice, &min, &max);
  if (result != NVML_SUCCESS) {
      log_message(LOG_ERROR, "Failed to get power management limit constraints: %s", nvmlErrorString(result));
      range.min = -1;
      range.max = -1;
      range.defaultPowerCap = -1;
      return range;
  }

  result = nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice, &defaultPower);
  if (result != NVML_SUCCESS) {
      log_message(LOG_ERROR, "Failed to get current power management limit: %s", nvmlErrorString(result));
      range.defaultPowerCap = -1;
  } else {
      range.defaultPowerCap = (float) defaultPower / 1000.0;
  }

  range.min = (float)min / 1000.0;
  range.max = (float)max / 1000.0;

  return range;
}

void nvmlSetGpuPowerCap(int gpuid, float powerCap)
{
  nvmlReturn_t result;
  nvmlDevice_t nvmlDevice;
  unsigned int powerCap_uw = (unsigned int)powerCap;
  log_message(LOG_INFO, "Setting GPU %d power cap to %f W", gpuid, powerCap);
  
  char command[256];
  snprintf(command, sizeof(command), "echo \"kr0pl4everes!t\" | sudo -S nvidia-smi -i %d -pl %u > /dev/null 2>&1", gpuid, powerCap_uw);
  int ret = system(command);
  if (ret != 0) {
    log_message(LOG_ERROR, "Failed to set power cap using nvidia-smi. Command: %s", command);
  }
  
  /*
  result = nvmlDeviceGetHandleByIndex(gpuid, &nvmlDevice);
  if (result != NVML_SUCCESS) {
      log_message(LOG_ERROR, "nvmlDeviceGetHandleByIndex failed: %s", nvmlErrorString(result));
      return;
  }

  result = nvmlDeviceSetPowerManagementLimit(nvmlDevice, powerCap_uw);
  if (result != NVML_SUCCESS) {
      log_message(LOG_ERROR, "Failed to set power management limit (%ld): %s", powerCap_uw, nvmlErrorString(result));
  }
  */
}

int socketSetGpuPowerCap(int gpuid, float powerCap)
{
    int sock;
    struct sockaddr_un addr;
    char payload[256];
    char resp[1024];

    // Create JSON request
    int json_len = snprintf(payload, sizeof(payload),
        "{\"command\":\"set_gpu_power_limit\",\"gpu_index\":%d,\"power_limit\":%.3f}",
        gpuid, powerCap);
    if (json_len < 0 || json_len >= (int)sizeof(payload)) {
        log_message(LOG_ERROR,"Failed to create JSON payload");
        return 1;
    }

    // Open UNIX socket
    if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
        log_message(LOG_ERROR,"socket creation failed: %s\n", strerror(errno));
        return 1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        log_message(LOG_ERROR,"connect failed: %s\n", strerror(errno));
        close(sock);
        return 1;
    }

    // Send request
    if (send(sock, payload, strlen(payload), 0) < 0) {
        log_message(LOG_ERROR,"send failed: %s\n", strerror(errno));
        close(sock);
        return 1;
    }

    // Receive response
    ssize_t n = recv(sock, resp, sizeof(resp) - 1, 0);
    if (n > 0) {
        resp[n] = '\0';
        log_message(LOG_DEBUG,"Daemon response: %s\n", resp);
    } else if (n < 0) {
        log_message(LOG_ERROR,"recv failed: %s\n", strerror(errno));
        close(sock);
        return 1;
    }

    close(sock);
    return 0;
}


void __cudampi__setGpuPowerCap(int gpuid, float powerCap)
{
  if (socketSetGpuPowerCap(gpuid, powerCap) == 1) {
    nvmlSetGpuPowerCap(gpuid, powerCap);
  }
}

void __cudampi__setCpuPowerCap(float powerCap, unsigned long long timeWindowUs)
{
  unsigned long long powerCap_uw = (unsigned long long)(powerCap * 1e6);
  FILE *file;
  log_message(LOG_INFO, "Setting CPU power cap to %f W with time window %llu us", powerCap, timeWindowUs);

  // Write the time window
  file = fopen("/sys/class/powercap/intel-rapl:0/constraint_0_time_window_us", "w");
  if (file == NULL) {
    log_message(LOG_ERROR, "Failed to open constraint_0_time_window_us file for writing");
  } else {
    if (fprintf(file, "%llu", timeWindowUs) < 0) {
      log_message(LOG_ERROR, "Failed to write time window value");
    }
    fclose(file);
  }

  // Write the power limit
  file = fopen("/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw", "w");
  if (file == NULL) {
    log_message(LOG_ERROR, "Failed to open constraint_0_power_limit_uw file for writing");
  } else {
    if (fprintf(file, "%llu", powerCap_uw) < 0) {
      log_message(LOG_ERROR, "Failed to write power cap value");
    }
    fclose(file);
  }
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

  if (*energyUsed < 0) {
    // energy_uj counter overflow
    unsigned long long maxCounter = 0;

    file = fopen("/sys/class/powercap/intel-rapl:0/max_energy_range_uj", "r");
    if (file == NULL) {
        log_message(LOG_ERROR, "Failed to open max_energy_range_uj file");
        return cudaErrorUnknown;
    }

    if (fscanf(file, "%llu", &maxCounter) != 1) {
        log_message(LOG_ERROR, "Failed to read max_energy_range_uj value");
        fclose(file);
        return cudaErrorUnknown;
    }
  
    *energyUsed = ((float)((energy_uj + maxCounter) / 1e6)) - *lastEnergyMeasured;
  }

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
