#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <nvml.h>
#include <omp.h>
#include <assert.h>
#include <math.h>

#define EDP_SCALING 480000

#define ENABLE_LOGGING
#define MPI_LOGGING
#include "logger.h"

#include "cudampicommon.h"
#include "cudampilib.h"
#include "powercap_config.h"
#include "powercap.h"
#include "cudampi_state.h"

powercapStrategy_t __cudampi__powercapStrategy = DISABLED;
perNodePowerCapRange_t* __cudampi__perNodePowerCapRange;
perNodePowerCapRange_t __cudampi__localPowerCapRange;
simpleGradientOpt_t __cudampi__simpleGradientOpt;
spsaGradientOpt_t __cudampi__spsaGradientOpt;
spsaAdaptiveOpt_t __cudampi__spsaAdaptiveOpt;
simpleAdaptiveOpt_t __cudampi__simpleAdaptiveOpt;
cmaesOpt_t __cudampi__cmaesOpt;
searchOpt_t __cudampi__searchOpt;
// Flag set by gradient optimizers when a full update (descent / CMA update) occurred
static int __cudampi__grad_update_performed = 0;

// Grid-search runtime parameters (configurable via powercap.conf). The grid spans
// the full normalized range with spacing derived from npoints, so spacing and its
// per-level decay are not free parameters (see __cudampi__gridSearchStep).
int __cudampi__search_depth = 3;
int __cudampi__search_grid_points = 5;

// Values below are expressed in terms of possible power cap range
// i.e. if min possible power cap is 100W and max is 250W, then 0.25 means 100W + 0.25 * (250W - 100W) = 137.5W
float __cudampi__cpu_min_powercap = 0.0;
float __cudampi__gpu_min_powercap = 0.0;
float __cudampi__gradient_opt_start_powercap = 1.0;

// CPU power cap time window (microseconds), broadcast to slaves
unsigned long long __cudampi__cpu_time_window_us = 1000000ULL; // default 1s
// Limit number of dynamic optimisation updates (0 = unlimited)
unsigned long long __cudampi__edp_optimization_steps = 0ULL;
// Number of sync windows to aggregate before each optimizer step (default 1)
unsigned long long __cudampi__optimizer_step_interval = 1ULL;

// Optional per-device starting power caps (fractions of each device's range,
// 0..1). When __cudampi__deviceCapOverrideCount > 0, these override the uniform
// start_powercap for EQUAL_SPLIT / EQUAL_SPLIT_EDP_MONITOR so a single device can
// be perturbed (used by the SNR diagnostic collector). 0 means "not specified".
float __cudampi__deviceCapOverride[__CUDAMPI_MAX_THREAD_COUNT];
int __cudampi__deviceCapOverrideCount = 0;

// Independent fixed cap fractions (0..1) for EDP_MONITOR_SPLIT. gpu_powercap is
// applied to every GPU device, cpu_powercap to every CPU device. -1 means unset
// (fall back to __cudampi__gradient_opt_start_powercap for that device type).
float __cudampi__gpu_powercap = -1.0f;
float __cudampi__cpu_powercap = -1.0f;

// Gradient optimisation runtime parameters (configurable via powercap.conf)
float __cudampi__gradient_start_alpha = 2.0f;
float __cudampi__gradient_alpha_decay = 0.99f;
float __cudampi__gradient_opt_eps = 5.0f;
float __cudampi__epsilon_decay = 0.99f;   // decay factor for epsilon (slow->fast schedule)

int first_sync_done = 0;
struct timeval prev_sync_time;
double __cudampi__edp = 0.0;

float getPowerCapFromRange(float min, float max, float target) {
  // if min = 100W and max = 250W, then target = 0.25 means  this function should return 100W + 0.25 * (250W - 100W) = 137.5W
  return min + target * (max - min);
}

float getFreePowerCap(int index) {
  return __cudampi__devicePowerConfig[index].powercapRange.max - __cudampi__devicePowerConfig[index].currentPowerCap;
}

static int __cudampi__isDynamicEdpStrategy(void) {
  return __cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE ||
         __cudampi__powercapStrategy == EDP_GRADIENT_SPSA ||
         __cudampi__powercapStrategy == EDP_GRADIENT_CMAES ||
         __cudampi__powercapStrategy == EDP_SEARCH_SHARED ||
         __cudampi__powercapStrategy == EDP_SEARCH_SPLIT;
}

static double __cudampi__clamp01(double v) {
  if (v < 0.0) return 0.0;
  if (v > 1.0) return 1.0;
  return v;
}

static double __cudampi__deviceLowerNorm(int i) {
  double lower = (i < __cudampi_totalgpudevicecount) ? __cudampi__gpu_min_powercap : __cudampi__cpu_min_powercap;
  return __cudampi__clamp01(lower);
}

static double __cudampi__clampNormForDevice(int i, double p) {
  double lower = __cudampi__deviceLowerNorm(i);
  if (p < lower) return lower;
  if (p > 1.0) return 1.0;
  return p;
}

void setDevicePowerCap(int index) {
  log_message(LOG_INFO, "Setting power cap of device %d to %.2f W", index, __cudampi__devicePowerConfig[index].currentPowerCap);
  if (index <  __cudampi__GPUcountspernode[0]) {
    __cudampi__setGpuPowerCap(index, __cudampi__devicePowerConfig[index].currentPowerCap);
  }
  else {
    MPI_Send(&__cudampi__devicePowerConfig[index].currentPowerCap, 1, MPI_FLOAT, 1, __cudampi__CONFIGUREPOWERCAP, __cudampi__communicators[index]);
  }
}

void __cudampi__resetLocalPowercaps() {
  if (__cudampi__powercapStrategy == DISABLED) {
    return;
  }
  // Reset CPU power cap
  __cudampi__setCpuPowerCap(__cudampi__localPowerCapRange.cpuRange.defaultPowerCap, __cudampi__localPowerCapRange.cpuRange.defaultTimeWindowUs);
  // Reset GPU power caps
  for (int i = 0; i < __cudampi__localGpuDeviceCount; i++) {
      __cudampi__setGpuPowerCap(i, __cudampi__localPowerCapRange.gpuRange[i].defaultPowerCap);
  }
}

void __cudampi__cmaesCleanup(void) {
  if (__cudampi__powercapStrategy == EDP_GRADIENT_CMAES) {
    cmaesOpt_t* g = &__cudampi__cmaesOpt;
    // Release CMA-ES resources if they were initialized
    if (g->n > 0) {
      cmaes_exit(&g->evo);
      cmaes_boundary_transformation_exit(&g->bounds);
      g->n = 0;
    }
  }
}

void __cudampi__updatePowerCap(float v, int i) {
  float minPowerCap = __cudampi__isDynamicEdpStrategy()
                    ? __cudampi__devicePowerConfig[i].minPowerCap
                    : __cudampi__devicePowerConfig[i].powercapRange.min;
  if (v < minPowerCap) {
    v = minPowerCap;
  } 
  else if (v > __cudampi__devicePowerConfig[i].powercapRange.max) {
    v = __cudampi__devicePowerConfig[i].powercapRange.max;
  }

  __cudampi__devicePowerConfig[i].currentPowerCap = v;
  setDevicePowerCap(i);
}

void __cudampi__applyAllPowercaps(void) {
  // Push configured power caps to devices based on selected strategy
  if (__cudampi__powercapStrategy == CONTINUOUS_GREEDY ||
      __cudampi__powercapStrategy == EQUAL_SHARE_CONTINUOUS_GREEDY ||
      __cudampi__powercapStrategy == EQUAL_SPLIT ||
      __cudampi__powercapStrategy == EQUAL_SPLIT_EDP_MONITOR ||
      __cudampi__powercapStrategy == EDP_MONITOR_SPLIT ||
      __cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE ||
      __cudampi__powercapStrategy == EDP_GRADIENT_SPSA ||
      __cudampi__powercapStrategy == EDP_GRADIENT_CMAES ||
      __cudampi__powercapStrategy == EDP_SEARCH_SHARED ||
      __cudampi__powercapStrategy == EDP_SEARCH_SPLIT) {
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      if (__cudampi__devicePowerConfig[i].currentPowerCap != -1) {
        setDevicePowerCap(i);
      }
    }
  }
}

float __cudampi__gettotalpowerofselecteddevices() { // gets total power of currently enabled devices
  int i;
  float power = 0;
  float curpower;

  omp_set_lock(&deviceselectionlock);

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_set_lock(&(__cudampi__devicelocks[i]));
  }

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].deviceEnabled == 1) {
      curpower = __cudampi__devicePowerConfig[i].currentPower;
      if (curpower == (-1)) {

        for (int i = 0; i < __cudampi_totaldevicecount; i++) {
          omp_unset_lock(&(__cudampi__devicelocks[i]));
        }

        omp_unset_lock(&deviceselectionlock);
        return -1;
      }
      power += curpower;
    }
  }

  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[i]));
  }

  omp_unset_lock(&deviceselectionlock);
  return power;
}

int __cudampi__selectpowercap_equal() { // adopts a greedy strategy for selecting devices
                                                     // returns 1 if successful, 0 otherwise - if not all devices have been recorder power
  int i;
  float powerleft;
  int indexselected;
  float curperfpower;
  int anydeviceenabled = 0;

  log_message(LOG_INFO, "---------- Executing equal power capping algorithm ! ---------- ");
  log_message(LOG_DEBUG, "\nBefore setting power cap");

  omp_set_lock(&deviceselectionlock);

  if (__cudampi__globalpowerlimit <= 0.0f) {
    log_message(LOG_DEBUG,"\nPowercap has not been set");
    omp_unset_lock(&deviceselectionlock);
    return 0;
  }

  powerleft = __cudampi__globalpowerlimit;
  // this will be invoked from one thread typically
  fflush(stdout);
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    log_message(LOG_DEBUG,"\nSetting lock on %d", i);
    fflush(stdout);
    omp_set_lock(&(__cudampi__devicelocks[i]));
    // disable all devices at first
  }
  fflush(stdout);

  // check of all the devices has been set power
  int allpowerset = 1;
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].currentPower == (-1)) {
      allpowerset = 0;
      break;
    }
  }

  if (!allpowerset) {
    // unlock and quit
    log_message(LOG_DEBUG,"Before setting powercap");
    fflush(stdout);
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }
    log_message(LOG_DEBUG,"After setting powercap");
    fflush(stdout);

    omp_unset_lock(&deviceselectionlock);

    return 0;
  }
  // All devices reported power, now select candidates for selection
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].deviceEnabled == 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = -1; // candidate for selection
    }
    __cudampi__amimanager[i] = 0;
  }

  fflush(stdout);
  int managerselected = 0;
  float totalFreeCapacity = 0;
  do {
    curperfpower = 0;
    indexselected = -1;
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      
      float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[i]) / __cudampi__devicePowerConfig[i].currentPowerCap;
      if (((-1) == (__cudampi__devicePowerConfig[i].deviceEnabled)) && (__cudampi__devicePowerConfig[i].minPowerCap <= powerleft) &&
          (inverseDeviceEnergyUsed > curperfpower)) {
        curperfpower = inverseDeviceEnergyUsed;
        indexselected = i;
        anydeviceenabled = 1;
      }
    }
    if (indexselected != (-1)) {
      anydeviceenabled = 1;
      // enable the found device now
      __cudampi__devicePowerConfig[indexselected].deviceEnabled = 1;
      if (!managerselected) {
        managerselected = 1;
        __cudampi__amimanager[indexselected] = 1;
      }

      __cudampi__devicePowerConfig[indexselected].currentPowerCap = __cudampi__devicePowerConfig[indexselected].minPowerCap;
      powerleft -= __cudampi__devicePowerConfig[indexselected].currentPowerCap;
      totalFreeCapacity +=  getFreePowerCap(indexselected);
      log_message(LOG_INFO,"Selected device %d", indexselected);
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
    if (__cudampi__devicePowerConfig[i].deviceEnabled != 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = 0;
    }
  }

  log_message(LOG_INFO, "All devices are selected and there is still power left: %f", powerleft);
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    if (!__cudampi__devicePowerConfig[i].deviceEnabled) {
      continue;
    }
    __cudampi__devicePowerConfig[i].currentPowerCap += powerleft * (getFreePowerCap(i) / totalFreeCapacity);
    log_message(LOG_INFO, "Device %d power cap increased to %f", i, __cudampi__devicePowerConfig[i].currentPowerCap);
    setDevicePowerCap(i);
  }

  // unlock the devices' locks
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[i]));
  }

  fflush(stdout);

  omp_unset_lock(&deviceselectionlock);

  return 1;
}


int __cudampi__SelectPowercapEqualEqualShare() {
  int i;
  float powerleft;
  int indexselected;
  float curperfpower;
  int anydeviceenabled = 0;
  int selectedGPUCount = 0;
  int selectedCPUCount = 0;
  const int totalCPUDevices = __cudampi_totaldevicecount - __cudampi_totalgpudevicecount;
  int deviceTypeToSelect = 1; // 0 -> CPU, 1 -> GPU (alternate)

  log_message(LOG_INFO, "---------- Executing equal power capping algorithm ! (with equal CPU and GPU share) ---------- ");
  log_message(LOG_DEBUG, "\nBefore setting power cap");

  omp_set_lock(&deviceselectionlock);

  if (__cudampi__globalpowerlimit <= 0.0f) {
    log_message(LOG_DEBUG,"\nPowercap has not been set");
    omp_unset_lock(&deviceselectionlock);
    return 0;
  }

  powerleft = __cudampi__globalpowerlimit;
  // this will be invoked from one thread typically
  fflush(stdout);
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    log_message(LOG_DEBUG,"\nSetting lock on %d", i);
    fflush(stdout);
    omp_set_lock(&(__cudampi__devicelocks[i]));
    // disable all devices at first
  }
  fflush(stdout);

  // check of all the devices has been set power
  int allpowerset = 1;
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].currentPower == (-1)) {
      allpowerset = 0;
      break;
    }
  }

  if (!allpowerset) {
    // unlock and quit
    log_message(LOG_DEBUG,"Before setting powercap");
    fflush(stdout);
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }
    log_message(LOG_DEBUG,"After setting powercap");
    fflush(stdout);

    omp_unset_lock(&deviceselectionlock);

    return 0;
  }
  // All devices reported power, now select candidates for selection
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].deviceEnabled == 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = -1; // candidate for selection
    }
    __cudampi__amimanager[i] = 0;
  }

  fflush(stdout);
  int managerselected = 0;
  float totalFreeCapacity = 0;
  do {
    curperfpower = 0;
    indexselected = -1;
    // If all CPUs are already selected, switch to GPU-only
    if (selectedCPUCount >= totalCPUDevices) {
      deviceTypeToSelect = 1; // GPU
    }

    // Consider only preferred type, maximize inverse energy used, and ensure minPowerCap fits
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      int isGPU = (i < __cudampi_totalgpudevicecount);
      int wantGPU = (deviceTypeToSelect == 1);
      if (__cudampi__devicePowerConfig[i].deviceEnabled != -1) continue;
      if (__cudampi__devicePowerConfig[i].minPowerCap > powerleft) continue;
      if (wantGPU && !isGPU) continue;
      if (!wantGPU && isGPU) continue;

      float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[i]) / __cudampi__devicePowerConfig[i].currentPowerCap;
      if (inverseDeviceEnergyUsed > curperfpower) {
        curperfpower = inverseDeviceEnergyUsed;
        indexselected = i;
      }
    }
    if (indexselected != (-1)) {
      // enable the found device now
      anydeviceenabled = 1;
      __cudampi__devicePowerConfig[indexselected].deviceEnabled = 1;
      if (!managerselected) {
        managerselected = 1;
        __cudampi__amimanager[indexselected] = 1;
      }

      __cudampi__devicePowerConfig[indexselected].currentPowerCap = __cudampi__devicePowerConfig[indexselected].minPowerCap;
      powerleft -= __cudampi__devicePowerConfig[indexselected].currentPowerCap;
      totalFreeCapacity +=  getFreePowerCap(indexselected);
      log_message(LOG_INFO,"Selected device %d", indexselected);
      log_message(LOG_INFO, "Remaining power left: %f", powerleft);

      if (indexselected >= __cudampi_totalgpudevicecount) {
        selectedCPUCount++;
        deviceTypeToSelect = 1; // next prefer GPU
      } else {
        selectedGPUCount++;
        deviceTypeToSelect = 0; // next prefer CPU
      }
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
    if (__cudampi__devicePowerConfig[i].deviceEnabled != 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = 0;
    }
  }

  log_message(LOG_INFO, "All devices are selected and there is still power left: %f", powerleft);
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    if (!__cudampi__devicePowerConfig[i].deviceEnabled) {
      continue;
    }
    __cudampi__devicePowerConfig[i].currentPowerCap += powerleft * (getFreePowerCap(i) / totalFreeCapacity);
    log_message(LOG_INFO, "Device %d power cap increased to %f", i, __cudampi__devicePowerConfig[i].currentPowerCap);
    setDevicePowerCap(i);
  }

  log_message(LOG_INFO, "Selected GPUs: %d", selectedGPUCount);
  log_message(LOG_INFO, "Selected CPUs: %d", selectedCPUCount);

  // unlock the devices' locks
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[i]));
  }

  fflush(stdout);

  omp_unset_lock(&deviceselectionlock);

  return 1;
}

//
// Fills g->delta[0..n-1] with a ±1 Rademacher vector and returns nothing.
// The vector is reused twice in the SPSA cycle: +eps*delta and -eps*delta.
//
static void __cudampi__generatePerturbation(spsaGradientOpt_t *g, int i)
{
  g->delta[i] = (rand() & 1) ? 1.0 : -1.0;   // uniform +- 1
}

//
// One call advances the Simultaneous-Perturbation Stochastic Approximation
// (SPSA) state machine by exactly one batch.  Two consecutive calls complete
// a full gradient-descent update, regardless of the number of devices.
//
// mode == 0   PLUS probe
//   • snapshot current caps into base_x
//   • build a new random perturbation delta
//   • write x_plus  = base_x + eps*delta      to all devices
//
// mode == 1   MINUS probe
//   • accumulate J_plus from the last batch
//   • write x_minus = base_x - eps*delta      to all devices
//
// mode == 2   DESCENT
//   • accumulate J_minus
//   • grad_i = (J_plus - J_minus)/(2*eps*delta_i)
//   • x_next = base_x - alpha*grad            (clamped in helper)
//   • write x_next and reset mode to 0
//
void __cudampi__gradientOptStepSpsa() {
  const int n = __cudampi_totaldevicecount;
  spsaGradientOpt_t* g = &__cudampi__spsaGradientOpt;
  // Stop only if alpha becomes negligibly small in normalized space
  if (g->alpha < 1e-6f) {
    log_message(LOG_INFO, "Alpha reached minimum threshold, stopping gradient optimisation");
    return;
  }

  switch (g->mode) {
    case PROBE_PLUS:
      log_message(LOG_INFO, "Gradient optimisation: entering PLUS_PROBE phase");

      for (int i = 0; i < n; ++i) {
        double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
        double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
        double pbase = __cudampi__cap_to_norm((double)__cudampi__devicePowerConfig[i].currentPowerCap, minv, maxv);
        pbase = __cudampi__clampNormForDevice(i, pbase);
        g->base_x[i] = pbase;

        __cudampi__generatePerturbation(g, i);
        double p = __cudampi__clampNormForDevice(i, pbase + (g->eps * g->delta[i]));
        g->x_plus[i] = p;
        double vcap = __cudampi__norm_to_cap(p, minv, maxv);

        log_message(LOG_INFO, "PROBE_PLUS: probe [%d] %f -> %f", i, g->base_x[i], p);
        __cudampi__updatePowerCap((float)vcap, i);
      }
      
      g->mode = PROBE_MINUS;
      return;
    case PROBE_MINUS:
      log_message(LOG_INFO, "Gradient optimisation: entering PROBE_MINUS phase");

      g->J_plus = __cudampi__edp;
      for (int i = 0; i < n; ++i) {
        double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
        double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
        double p = __cudampi__clampNormForDevice(i, g->base_x[i] - (g->eps * g->delta[i]));
        g->x_minus[i] = p;
        double vcap = __cudampi__norm_to_cap(p, minv, maxv);

        log_message(LOG_INFO, "PROBE_MINUS: probe [%d] %f -> %f", i, g->base_x[i], p);
        __cudampi__updatePowerCap((float)vcap, i);
      }
      
      g->mode = DESCENT;
      return;
    case DESCENT:
      log_message(LOG_INFO, "Gradient optimisation: entering DESCENT phase");

      double J_minus = __cudampi__edp;
      log_message(LOG_INFO, "DESCENT: J_plus=%lf J_minus=%lf", g->J_plus, J_minus);
      for (int i = 0; i < n; ++i) {
        // log-objective SPSA gradient in normalized space
        double denom = g->x_plus[i] - g->x_minus[i];
        double grad = 0.0;
        if (fabs(denom) >= 1e-6) {
          grad = (log(g->J_plus) - log(J_minus)) / denom;
        }
        double p = __cudampi__clampNormForDevice(i, g->base_x[i] - (g->alpha * grad));
        double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
        double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
        double vcap = __cudampi__norm_to_cap(p, minv, maxv);

        log_message(LOG_INFO, "SPSA grad[%d]: delta=%.0f x_minus=%.6f x_plus=%.6f denom=%.6f J+=%.8f J-=%.8f grad=%.8f",
                    i, g->delta[i], g->x_minus[i], g->x_plus[i], denom, g->J_plus, J_minus, grad);
        log_message(LOG_INFO, "DESCENT: probe [%d] %f -> %f with grad=%lf", i, g->base_x[i], p, grad);
        __cudampi__updatePowerCap((float)vcap, i);
      }
      // Mark that a full gradient update has been performed
      __cudampi__grad_update_performed = 1;
      
      g->alpha *= __cudampi__gradient_alpha_decay;
      // Update epsilon with slow-then-fast decay: eps = eps0 * (decay)^(iter^2)
      g->iter += 1;
      g->eps = g->eps0 * pow((double)__cudampi__epsilon_decay, (double)(g->iter));
      log_message(LOG_INFO, "SPSA: iter=%d eps=%.6f alpha=%.6f", g->iter, g->eps, g->alpha);
      g->mode = PROBE_PLUS;
      return;
    default:
      log_message(LOG_ERROR, "Gradient optimisation: invalid mode %d", g->mode);
      return;
  }
}

/* CMA-ES boundary handling is done via cmaes_boundary_transformation.
 * The previous custom sigmoid/logit mapping is superseded by that. */

void __cudampi__gradientOptStepCmaes() {
  cmaesOpt_t* g = &__cudampi__cmaesOpt;
  const int n = g->n;

  if (g->terminated) {
    // Do not change power caps anymore after termination
    return;
  }

  if (g->mode == CMA_SAMPLE) {
    log_message(LOG_INFO, "CMA-ES: SAMPLE population");
    g->pop = cmaes_SamplePopulation(&g->evo); // returns [lambda][n]
    g->k = 0;
    // write first candidate caps (transform internal -> normalized [0,1] -> physical)
    cmaes_boundary_transformation(&g->bounds, g->pop[g->k], g->x_in_bounds, n);
    for (int i = 0; i < n; ++i) {
      double vcap = __cudampi__norm_to_cap(g->x_in_bounds[i], g->phys_minv[i], g->phys_maxv[i]);
      __cudampi__updatePowerCap((float)vcap, i);
    }
    g->mode = CMA_EVAL;
    return;
  }

  if (g->mode == CMA_EVAL) {
    log_message(LOG_INFO, "CMA-ES: EVAL");
    // compute fitness for candidate k
    double J = __cudampi__edp;

    // Log candidate fitness to inspect fluctuations
    log_message(LOG_INFO, "CMA-ES: candidate %d/%d fitness J = %.8f", g->k, g->lambda, J);
    g->f[g->k] = J;
    g->k += 1;
    if (g->k < g->lambda) {
      // schedule next candidate
      cmaes_boundary_transformation(&g->bounds, g->pop[g->k], g->x_in_bounds, n);
      for (int i = 0; i < n; ++i) {
        double vcap = __cudampi__norm_to_cap(g->x_in_bounds[i], g->phys_minv[i], g->phys_maxv[i]);
        __cudampi__updatePowerCap((float)vcap, i);
      }
      return; // remain in EVAL until last
    }
    g->mode = CMA_UPDATE;
    return;
  }

  if (g->mode == CMA_UPDATE) {
    log_message(LOG_INFO, "CMA-ES: UPDATE distribution");
    // Report per-generation fitness statistics before updating the distribution
    if (g->lambda > 0) {
      double minJ = g->f[0], maxJ = g->f[0];
      double sumJ = 0.0, sumJ2 = 0.0;
      for (int i = 0; i < g->lambda; ++i) {
        double v = g->f[i];
        if (v < minJ) minJ = v;
        if (v > maxJ) maxJ = v;
        sumJ += v;
        sumJ2 += v * v;
      }
      double meanJ = sumJ / (double)g->lambda;
      double varJ = (sumJ2 / (double)g->lambda) - (meanJ * meanJ);
      if (varJ < 0.0) varJ = 0.0;
      double sdJ = sqrt(varJ);
      long gen = (long)cmaes_Get(&g->evo, "gen");
      log_message(LOG_INFO, "CMA-ES: gen %ld fitness stats: min=%.8f mean=%.8f sd=%.8f max=%.8f", gen, minJ, meanJ, sdJ, maxJ);
      // Print current sigma and best-ever solution (x and J) if available
      double sigma = cmaes_Get(&g->evo, "sigma");
      double fbest = cmaes_Get(&g->evo, "fbestever");
      const double* xbest = cmaes_GetPtr(&g->evo, "xbestever");
      // Map best-ever internal solution to normalized and physical caps
      if (xbest != NULL) {
        cmaes_boundary_transformation(&g->bounds, xbest, g->x_in_bounds, n);
        log_message(LOG_INFO, "CMA-ES: sigma=%.6g fbest=%.8f", sigma, fbest);
        for (int i = 0; i < n; ++i) {
          double cap = __cudampi__norm_to_cap(g->x_in_bounds[i], g->phys_minv[i], g->phys_maxv[i]);
          log_message(LOG_INFO, "CMA-ES: best x[%d]=%.6f (cap=%.3f)", i, g->x_in_bounds[i], cap);
        }
      } else {
        log_message(LOG_INFO, "CMA-ES: sigma=%.6g (no xbestever yet)", sigma);
      }
    }
    cmaes_UpdateDistribution(&g->evo, g->f);
    // Count this as one CMA-ES update phase
    __cudampi__grad_update_performed = 1;
    // No termination check; continue iterating until external step limit
    // Optionally set to current mean for stability between generations
    cmaes_boundary_transformation(&g->bounds, g->evo.rgxbestever, g->x_in_bounds, n);
    for (int i = 0; i < n; ++i) {
      double vcap = __cudampi__norm_to_cap(g->x_in_bounds[i], g->phys_minv[i], g->phys_maxv[i]);
      __cudampi__updatePowerCap((float)vcap, i);
    }
    g->mode = CMA_SAMPLE;
    return;
  }

  log_message(LOG_ERROR, "CMA-ES: invalid mode %d", g->mode);
}
  
void __cudampi__gradientOptStepSimple() {
  const int n = __cudampi_totaldevicecount;
  simpleGradientOpt_t* g = &__cudampi__simpleGradientOpt;

  // BASE phase: capture reference sample and continue to PROBE_PLUS
  if (g->mode == BASE) {
    log_message(LOG_INFO, "Gradient optimisation: entering BASE phase");
    g->base_y = __cudampi__edp;
    for (int i = 0; i < n; ++i) {
      double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
      double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
      double p = __cudampi__cap_to_norm((double)__cudampi__devicePowerConfig[i].currentPowerCap, minv, maxv);
      p = __cudampi__clampNormForDevice(i, p);
      g->base_x[i] = p;
    }
    g->probe_dim = 0;
    g->mode = PROBE_PLUS;  // schedule first +eps in PROBE_PLUS below
  }

  // PROBE_PLUS: save J_minus for previous dim (if any), schedule +eps for current dim
  if (g->mode == PROBE_PLUS) {
    if (g->probe_dim > 0) {
      int dprev = g->probe_dim - 1;
      double J_minus_prev = __cudampi__edp;
      double denom = g->x_plus[dprev] - g->x_minus[dprev];
      g->grad[dprev] = 0.0;
      if (fabs(denom) >= 1e-6) {
        g->grad[dprev] = (log(g->J_plus) - log(J_minus_prev)) / denom;
      }
      log_message(LOG_INFO, "FD grad[%d]: x_minus=%.6f x_plus=%.6f denom=%.6f J+=%.8f J-=%.8f grad=%.8f",
                  dprev, g->x_minus[dprev], g->x_plus[dprev], denom, g->J_plus, J_minus_prev, g->grad[dprev]);
    }
    // schedule +eps probe for current coordinate
    for (int i = 0; i < n; ++i) {
      double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
      double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
      double p = g->base_x[i];
      if (i == g->probe_dim) {
        p = __cudampi__clampNormForDevice(i, g->base_x[i] + g->eps);
        g->x_plus[i] = p;
      }
      double vcap = __cudampi__norm_to_cap(p, minv, maxv);
      log_message(LOG_INFO, "FD-CENTRAL: PLUS probe [%d] %f -> %f", i, g->base_x[i], p);
      __cudampi__updatePowerCap((float)vcap, i);
    }
    g->mode = PROBE_MINUS; // next call: record J_plus
    return;
  }

  else if (g->mode == PROBE_MINUS) {
    // just completed +eps probe for current dim: store J_plus
    g->J_plus = __cudampi__edp;
    // schedule -eps probe for the same coordinate
    for (int i = 0; i < n; ++i) {
      double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
      double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
      double p = g->base_x[i];
      if (i == g->probe_dim) {
        p = __cudampi__clampNormForDevice(i, g->base_x[i] - g->eps);
        g->x_minus[i] = p;
      }
      double vcap = __cudampi__norm_to_cap(p, minv, maxv);
      log_message(LOG_INFO, "FD-CENTRAL: MINUS probe [%d] %f -> %f", i, g->base_x[i], p);
      __cudampi__updatePowerCap((float)vcap, i);
    }
    if (g->probe_dim < n - 1) {
      g->probe_dim += 1;   // move to next dimension
      g->mode = PROBE_PLUS;
    } else {
      g->mode = DESCENT;   // last dimension: next call will compute grad and descend
    }
    return;
  }
  else if (g->mode == DESCENT) {
    // capture J_minus for the last dimension and compute its gradient
    int last = n - 1;
    double J_minus_last = __cudampi__edp;
    double denom = g->x_plus[last] - g->x_minus[last];
    g->grad[last] = 0.0;
    if (fabs(denom) >= 1e-6) {
      g->grad[last] = (log(g->J_plus) - log(J_minus_last)) / denom;
    }
    log_message(LOG_INFO, "FD grad[%d]: x_minus=%.6f x_plus=%.6f denom=%.6f J+=%.8f J-=%.8f grad=%.8f",
                last, g->x_minus[last], g->x_plus[last], denom, g->J_plus, J_minus_last, g->grad[last]);

    // full gradient available: take one descent step
    for (int i = 0; i < n; ++i) {
      double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
      double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
      double p = __cudampi__clampNormForDevice(i, g->base_x[i] - g->alpha * g->grad[i]);
      double vcap = __cudampi__norm_to_cap(p, minv, maxv);
      log_message(LOG_INFO, "FD-CENTRAL: descent [%d] %f -> %f", i, g->base_x[i], p);
      __cudampi__updatePowerCap((float)vcap, i);  // helper clamps to limits
    }

    // Mark that a full gradient update has been performed
    __cudampi__grad_update_performed = 1;
    g->alpha *= __cudampi__gradient_alpha_decay;
    // Update epsilon with slow-then-fast decay: eps = eps0 * (decay)^(iter^2)
    g->iter += 1;
    g->eps = g->eps0 * pow((double)__cudampi__epsilon_decay, (double)(g->iter));
    log_message(LOG_INFO, "FD-CENTRAL: iter=%d eps=%.6f", g->iter, g->eps);
    g->mode = BASE;  // restart cycle with a new base sample next time
  }
}

//
// Grid-search EDP minimisation.
//
// The search evaluates a grid of power-cap settings sequentially (one grid point
// per optimiser step, reusing the same one-call lag as the gradient strategies:
// each call reads __cudampi__edp produced by the caps applied on the previous
// call).
//
// Level 0 spans the full normalized range [0,1] with npoints equally spaced
// points (spacing = 1/npoints, centred at 0.5): e.g. npoints=5 -> 0.1,0.3,..,0.9.
// After a level is evaluated, the best (lowest-EDP) point becomes the new centre
// and the interval between its two neighbours is subdivided into npoints interior
// points (endpoints excluded) for the next level. That subdivision is exactly a
// spacing shrink by 2/(npoints+1), so the grid resolution is the only knob; there
// is no separate eps / eps_decay. After search_depth levels the best caps are held.
//
// dims == 1 (EDP_SEARCH_SHARED): a single fraction applied to all devices.
// dims == 2 (EDP_SEARCH_SPLIT):  axis[0] = GPU fraction, axis[1] = CPU fraction.
//

// Map a flat grid index k to per-axis normalized coordinates around the centre.
static void __cudampi__searchAxisForK(searchOpt_t* g, int k, double out[2]) {
  double half = (g->npoints - 1) / 2.0;
  if (g->dims == 1) {
    out[0] = g->center[0] + ((double)k - half) * g->spacing;
    out[1] = out[0];
  } else {
    int idxGpu = k % g->npoints;
    int idxCpu = k / g->npoints;
    out[0] = g->center[0] + ((double)idxGpu - half) * g->spacing;
    out[1] = g->center[1] + ((double)idxCpu - half) * g->spacing;
  }
}

// Push the given per-axis fractions to all devices (clamped per device type).
static void __cudampi__searchApplyAxis(searchOpt_t* g, const double axis[2]) {
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    int isGpu = (i < __cudampi_totalgpudevicecount);
    double a = (g->dims == 1) ? axis[0] : (isGpu ? axis[0] : axis[1]);
    double p = __cudampi__clampNormForDevice(i, a);
    double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
    double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
    double vcap = __cudampi__norm_to_cap(p, minv, maxv);
    __cudampi__updatePowerCap((float)vcap, i);
  }
}

static void __cudampi__searchApplyK(searchOpt_t* g, int k) {
  double axis[2];
  __cudampi__searchAxisForK(g, k, axis);
  if (g->dims == 1) {
    log_message(LOG_INFO, "SEARCH: level=%d point %d/%d shared=%.4f spacing=%.4f",
                g->level, k, g->total, axis[0], g->spacing);
  } else {
    log_message(LOG_INFO, "SEARCH: level=%d point %d/%d gpu=%.4f cpu=%.4f spacing=%.4f",
                g->level, k, g->total, axis[0], axis[1], g->spacing);
  }
  __cudampi__searchApplyAxis(g, axis);
}

// Set grid point-0 caps as each device's initial currentPowerCap WITHOUT pushing
// them to the devices. Used at init time only: __cudampi__applyInitialPowercapsForStrategy
// runs before the MPI communicators are allocated, so a setDevicePowerCap() MPI_Send
// to a slave would dereference a null communicator. The deferred push happens in
// __cudampi__applyAllPowercaps(), exactly like the gradient strategies.
static void __cudampi__searchInitCaps(searchOpt_t* g) {
  double axis[2];
  __cudampi__searchAxisForK(g, 0, axis);
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    int isGpu = (i < __cudampi_totalgpudevicecount);
    double a = (g->dims == 1) ? axis[0] : (isGpu ? axis[0] : axis[1]);
    double p = __cudampi__clampNormForDevice(i, a);
    double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
    double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
    double vcap = __cudampi__norm_to_cap(p, minv, maxv);
    // Match __cudampi__updatePowerCap's dynamic-strategy clamp (floor at minPowerCap).
    if (vcap < __cudampi__devicePowerConfig[i].minPowerCap) {
      vcap = __cudampi__devicePowerConfig[i].minPowerCap;
    } else if (vcap > __cudampi__devicePowerConfig[i].powercapRange.max) {
      vcap = __cudampi__devicePowerConfig[i].powercapRange.max;
    }
    __cudampi__devicePowerConfig[i].currentPowerCap = (float)vcap;
  }
}

void __cudampi__gridSearchStep() {
  searchOpt_t* g = &__cudampi__searchOpt;

  if (g->terminated || g->mode == SEARCH_DONE) {
    // Best caps already held; nothing more to change.
    return;
  }

  // mode == SEARCH_EVAL: __cudampi__edp is the result of grid point g->k.
  double J = __cudampi__edp;
  double axis[2];
  __cudampi__searchAxisForK(g, g->k, axis);
  log_message(LOG_INFO, "SEARCH: eval level=%d point %d/%d J=%.8f (gpu=%.4f cpu=%.4f)",
              g->level, g->k, g->total, J, axis[0], axis[1]);

  if (J > 0.0 && (!g->have_best || J < g->bestJ)) {
    g->bestJ = J;
    g->best_axis[0] = axis[0];
    g->best_axis[1] = axis[1];
    g->have_best = 1;
  }

  g->k += 1;
  if (g->k < g->total) {
    __cudampi__searchApplyK(g, g->k);
    return;
  }

  // Level complete: recentre on the best point found.
  if (g->have_best) {
    g->center[0] = __cudampi__clamp01(g->best_axis[0]);
    g->center[1] = __cudampi__clamp01(g->best_axis[1]);
  }
  log_message(LOG_INFO, "SEARCH: level=%d complete bestJ=%.8f centre gpu=%.4f cpu=%.4f",
              g->level, g->bestJ, g->center[0], g->center[1]);
  // Count one completed refinement level as an optimisation update.
  __cudampi__grad_update_performed = 1;

  g->level += 1;
  if (g->level >= g->depth) {
    // Max depth reached: hold the best caps and stop.
    log_message(LOG_INFO, "SEARCH: max depth %d reached, holding best caps (gpu=%.4f cpu=%.4f J=%.8f)",
                g->depth, g->center[0], g->center[1], g->bestJ);
    g->terminated = 1;
    g->mode = SEARCH_DONE;
    __cudampi__searchApplyAxis(g, g->center);
    return;
  }

  // Refine: subdivide the interval between the best point's two neighbours into
  // npoints interior points around the new centre. That is a spacing shrink of
  // 2/(npoints+1) (e.g. npoints=5: 0.2 -> 0.0667, so a best of 0.3 is searched
  // over 0.1667..0.4333, strictly inside its old neighbours 0.1 and 0.5).
  g->spacing *= 2.0 / (double)(g->npoints + 1);
  g->have_best = 0;
  g->k = 0;
  __cudampi__searchApplyK(g, 0);
}

int __cudampi__selectpowercap_gradient() {
  omp_set_lock(&deviceselectionlock);
 
  int allpowerset = 1;
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].currentPower == (-1)) {
      omp_unset_lock(&deviceselectionlock);
      return 0;
    }
  }

  if (__cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE) {
    __cudampi__gradientOptStepSimple();
  } else if (__cudampi__powercapStrategy == EDP_GRADIENT_SPSA) {
    __cudampi__gradientOptStepSpsa();
  } else if (__cudampi__powercapStrategy == EDP_GRADIENT_CMAES) {
    __cudampi__gradientOptStepCmaes();
  } else if (__cudampi__powercapStrategy == EDP_SEARCH_SHARED ||
             __cudampi__powercapStrategy == EDP_SEARCH_SPLIT) {
    __cudampi__gridSearchStep();
  }

  omp_unset_lock(&deviceselectionlock);

  return 1;
}

int __cudampi__selectdevicesforpowerlimit_greedy() { // adopts a greedy strategy for selecting devices
                                                     // returns 1 if successful, 0 otherwise - if not all devices have been recorder power
  int i;
  float powerleft;
  int indexselected;
  float curperfpower;
  int anydeviceenabled = 0;

  log_message(LOG_INFO, "---------- Executing greedy power capping algorithm ! ---------- ");
  fflush(stdout);

  omp_set_lock(&deviceselectionlock);

  if (__cudampi__globalpowerlimit <= 0.0f) {
    log_message(LOG_DEBUG,"\nPowercap has not been set");
    fflush(stdout);
    omp_unset_lock(&deviceselectionlock);
    return 0;
  }

  powerleft = __cudampi__globalpowerlimit;
  // this will be invoked from one thread typically
  fflush(stdout);
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    log_message(LOG_DEBUG,"\nSetting lock on %d", i);
    fflush(stdout);
    omp_set_lock(&(__cudampi__devicelocks[i]));
    // disable all devices at first
  }
  fflush(stdout);

  // check of all the devices has been set power
  int allpowerset = 1;
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].currentPower == (-1)) {
      allpowerset = 0;
      break;
    }
  }

  if (!allpowerset) {
    // unlock and quit
    log_message(LOG_DEBUG,"Before setting powercap");
    fflush(stdout);
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }
    log_message(LOG_DEBUG,"After setting powercap");
    fflush(stdout);

    omp_unset_lock(&deviceselectionlock);

    return 0;
  }
  // All devices reported power, now select candidates for selection
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].deviceEnabled == 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = -1; // candidate for selection
    }
    __cudampi__amimanager[i] = 0;
  }

  fflush(stdout);
  int managerselected = 0;
  do {
    curperfpower = 0;
    indexselected = -1;
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      
      float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[i]) / __cudampi__devicePowerConfig[i].currentPower;
      if (((-1) == (__cudampi__devicePowerConfig[i].deviceEnabled)) && (__cudampi__devicePowerConfig[i].currentPower <= powerleft) &&
          (inverseDeviceEnergyUsed > curperfpower)) {
        curperfpower = inverseDeviceEnergyUsed;
        indexselected = i;
        anydeviceenabled = 1;
      }
    }
    if (indexselected != (-1)) {
      // enable the found device now
      __cudampi__devicePowerConfig[indexselected].deviceEnabled = 1;
      if (!managerselected) {
        managerselected = 1;
        __cudampi__amimanager[indexselected] = 1;
      }

      // Log if the selected device is CPU or GPU and how much power it subtracts
      if (indexselected >= __cudampi_totalgpudevicecount) {
        if(__cudampi__devicePowerConfig[indexselected].currentPower <= 0)
        {
          log_message(LOG_ERROR,"Selected CPU device %d is not correctly calculated", indexselected);
        }
        log_message(LOG_INFO, "Selected CPU device %d, subtracted power: %f, inverse device energy used: %f", indexselected, __cudampi__devicePowerConfig[indexselected].currentPower, curperfpower);
      } else {
        log_message(LOG_INFO, "Selected GPU device %d, subtracted power: %f, inverse device energy used: %f", indexselected, __cudampi__devicePowerConfig[indexselected].currentPower, curperfpower);
      }

      // Log devices that were not selected and their power
      for (int j = 0; j < __cudampi_totaldevicecount; j++) {
        if (__cudampi__devicePowerConfig[j].deviceEnabled != 1) {
          float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[j]) / __cudampi__devicePowerConfig[j].currentPower;
          if (j >= __cudampi_totalgpudevicecount) {
        log_message(LOG_INFO, "Not selected CPU device %d, power: %f, inverse device energy used: %f", j, __cudampi__devicePowerConfig[j].currentPower, inverseDeviceEnergyUsed);
          } else {
        log_message(LOG_INFO, "Not selected GPU device %d, power: %f, inverse device energy used: %f", j, __cudampi__devicePowerConfig[j].currentPower, inverseDeviceEnergyUsed);
          }
        }
      }


      powerleft -= __cudampi__devicePowerConfig[indexselected].currentPower;
      log_message(LOG_INFO,"\nSelected device %d", indexselected);
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
    if (__cudampi__devicePowerConfig[i].deviceEnabled != 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = 0;
    }
  }

   if (powerleft > 0) {
    log_message(LOG_INFO, "All devices are selected and there is still power left: %f", powerleft);
  }

  // unlock the devices' locks
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[i]));
  }

  fflush(stdout);

  omp_unset_lock(&deviceselectionlock);

  return 1;
}



int __cudampi__selectDevicesForPowerlimitGreedyEqualShare() { 
  int i;
  float powerleft;
  int indexselected;
  float curperfpower;
  int anydeviceenabled = 0;
  int selectedGPUCount = 0;
  int selectedCPUCount = 0;
  const int totalCPUDevices = __cudampi_totaldevicecount - __cudampi_totalgpudevicecount;
  int deviceTypeToSelect = 1; // 0 -> CPU, 1 -> GPU

  log_message(LOG_INFO, "---------- Executing greedy power capping algorithm (with equal CPU and GPU share) ! ---------- ");
  fflush(stdout);

  omp_set_lock(&deviceselectionlock);

  if (__cudampi__globalpowerlimit <= 0.0f) {
    log_message(LOG_DEBUG,"\nPowercap has not been set");
    fflush(stdout);
    omp_unset_lock(&deviceselectionlock);
    return 0;
  }

  powerleft = __cudampi__globalpowerlimit;
  // this will be invoked from one thread typically
  fflush(stdout);
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    log_message(LOG_DEBUG,"\nSetting lock on %d", i);
    fflush(stdout);
    omp_set_lock(&(__cudampi__devicelocks[i]));
    // disable all devices at first
  }
  fflush(stdout);

  // check of all the devices has been set power
  int allpowerset = 1;
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].currentPower == (-1)) {
      allpowerset = 0;
      break;
    }
  }

  if (!allpowerset) {
    // unlock and quit
    log_message(LOG_DEBUG,"Before setting powercap");
    fflush(stdout);
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }
    log_message(LOG_DEBUG,"After setting powercap");
    fflush(stdout);

    omp_unset_lock(&deviceselectionlock);

    return 0;
  }
  // All devices reported power, now select candidates for selection
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].deviceEnabled == 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = -1; // candidate for selection
    }
    __cudampi__amimanager[i] = 0;
  }

  fflush(stdout);
  int managerselected = 0;
  do {
    curperfpower = 0;
    indexselected = -1;

    // If all CPUs are already selected, switch to GPUs only
    if (selectedCPUCount >= totalCPUDevices) {
      deviceTypeToSelect = 1; // GPU
    }

    // Try preferred type only  maximize inverse energy used
    for (i = 0; i < __cudampi_totaldevicecount; i++) {
      int isGPU = (i < __cudampi_totalgpudevicecount);
      int wantGPU = (deviceTypeToSelect == 1);
      if (((-1) != __cudampi__devicePowerConfig[i].deviceEnabled) ||
          (__cudampi__devicePowerConfig[i].currentPower > powerleft)) {
        continue;
      }
      if (wantGPU && !isGPU) continue;
      if (!wantGPU && isGPU) continue;
      float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[i]) / __cudampi__devicePowerConfig[i].currentPower;
      if (inverseDeviceEnergyUsed > curperfpower) {
        curperfpower = inverseDeviceEnergyUsed;
        indexselected = i;
        anydeviceenabled = 1;
      }
    }
    if (indexselected != (-1)) {
      // enable the found device now
      __cudampi__devicePowerConfig[indexselected].deviceEnabled = 1;
      if (!managerselected) {
        managerselected = 1;
        __cudampi__amimanager[indexselected] = 1;
      }

      // Log if the selected device is CPU or GPU and how much power it subtracts
      if (indexselected >= __cudampi_totalgpudevicecount) {
        if(__cudampi__devicePowerConfig[indexselected].currentPower <= 0)
        {
          log_message(LOG_ERROR,"Selected CPU device %d is not correctly calculated", indexselected);
        }
        log_message(LOG_INFO, "Selected CPU device %d, subtracted power: %f, inverse device energy used: %f", indexselected, __cudampi__devicePowerConfig[indexselected].currentPower, curperfpower);
        selectedCPUCount++;
        deviceTypeToSelect = 1; // next time prefer GPU
      } else {
        log_message(LOG_INFO, "Selected GPU device %d, subtracted power: %f, inverse device energy used: %f", indexselected, __cudampi__devicePowerConfig[indexselected].currentPower, curperfpower);
        selectedGPUCount++;
        deviceTypeToSelect = 0; // next time prefer CPU
      }

      // Log devices that were not selected and their power
      for (int j = 0; j < __cudampi_totaldevicecount; j++) {
        if (__cudampi__devicePowerConfig[j].deviceEnabled != 1) {
          float inverseDeviceEnergyUsed = computeDevPerformance(__cudampi__time_us[j]) / __cudampi__devicePowerConfig[j].currentPower;
          if (j >= __cudampi_totalgpudevicecount) {
        log_message(LOG_INFO, "Not selected CPU device %d, power: %f, inverse device energy used: %f", j, __cudampi__devicePowerConfig[j].currentPower, inverseDeviceEnergyUsed);
          } else {
        log_message(LOG_INFO, "Not selected GPU device %d, power: %f, inverse device energy used: %f", j, __cudampi__devicePowerConfig[j].currentPower, inverseDeviceEnergyUsed);
          }
        }
      }


      powerleft -= __cudampi__devicePowerConfig[indexselected].currentPower;
      log_message(LOG_INFO,"\nSelected device %d", indexselected);
      log_message(LOG_INFO, "Remaining power left: %f", powerleft);
    }
  } while (indexselected != (-1));
  fflush(stdout);

  if (!anydeviceenabled) { // handle this case
    log_message(LOG_ERROR,"No devices found under the power limit");
    fflush(stdout);
    exit(-1);
  }

  // Ensure at least one CPU and one GPU selected
  if (selectedCPUCount == 0 || selectedGPUCount == 0) {
    log_message(LOG_ERROR, "Could not select at least one CPU and one GPU under the power limit (GPUs=%d, CPUs=%d)", selectedGPUCount, selectedCPUCount);
  }

  // now not enabled devices are set to 0
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    if (__cudampi__devicePowerConfig[i].deviceEnabled != 1) {
      __cudampi__devicePowerConfig[i].deviceEnabled = 0;
    }
  }

   if (powerleft > 0) {
    log_message(LOG_INFO, "All devices are selected and there is still power left: %f", powerleft);
  }

  log_message(LOG_INFO, "Selected GPUs: %d", selectedGPUCount);
  log_message(LOG_INFO, "Selected CPUs: %d", selectedCPUCount);

  // unlock the devices' locks
  for (i = 0; i < __cudampi_totaldevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[i]));
  }

  fflush(stdout);

  omp_unset_lock(&deviceselectionlock);

  return 1;
}

void __cudampi__initializeGradientOpt (double alpha, double eps) {
  // For simplicity initialize both optimisers, but only one will be used
  // Interpret eps in normalized [0,1] power-cap space; clamp to a sane range
  double neps = eps;
  if (neps <= 0.0) neps = 0.02; // default small step in normalized units
  if (neps > 0.5) neps = 0.5;

  __cudampi__simpleGradientOpt.alpha = alpha;
  __cudampi__simpleGradientOpt.eps = neps;
  __cudampi__simpleGradientOpt.eps0 = neps;
  __cudampi__simpleGradientOpt.mode = BASE;
  __cudampi__simpleGradientOpt.iter = 0;
  __cudampi__spsaGradientOpt.alpha = alpha;
  __cudampi__spsaGradientOpt.eps = neps;
  __cudampi__spsaGradientOpt.eps0 = neps;
  __cudampi__spsaGradientOpt.mode = PROBE_PLUS;
  __cudampi__spsaGradientOpt.iter = 0;
  // Adaptive (Adam + SPSA) initialisation
  __cudampi__spsaAdaptiveOpt.alpha = alpha;
  __cudampi__spsaAdaptiveOpt.eps = neps;
  __cudampi__spsaAdaptiveOpt.eps0 = neps;
  __cudampi__spsaAdaptiveOpt.beta1 = 0.9;
  __cudampi__spsaAdaptiveOpt.beta2 = 0.999;
  __cudampi__spsaAdaptiveOpt.adam_eps = 1e-8;
  __cudampi__spsaAdaptiveOpt.t = 0;
  __cudampi__spsaAdaptiveOpt.mode = PROBE_PLUS;
  // Simple Adaptive (Adam + FD) initialisation
  __cudampi__simpleAdaptiveOpt.alpha = alpha;
  __cudampi__simpleAdaptiveOpt.eps = neps;
  __cudampi__simpleAdaptiveOpt.eps0 = neps;
  __cudampi__simpleAdaptiveOpt.beta1 = 0.9;
  __cudampi__simpleAdaptiveOpt.beta2 = 0.999;
  __cudampi__simpleAdaptiveOpt.adam_eps = 1e-8;
  __cudampi__simpleAdaptiveOpt.t = 0;
  __cudampi__simpleAdaptiveOpt.mode = BASE;
  for (int i = 0; i < __cudampi_totaldevicecount; ++i) {
    double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
    double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
    double pnow = __cudampi__cap_to_norm((double)__cudampi__devicePowerConfig[i].currentPowerCap, minv, maxv);
    pnow = __cudampi__clampNormForDevice(i, pnow);
    __cudampi__simpleGradientOpt.base_x[i]  = pnow;
    __cudampi__simpleGradientOpt.x_plus[i]  = pnow;
    __cudampi__simpleGradientOpt.x_minus[i] = pnow;
    __cudampi__spsaGradientOpt.base_x[i]    = pnow;
    __cudampi__spsaGradientOpt.x_plus[i]    = pnow;
    __cudampi__spsaGradientOpt.x_minus[i]   = pnow;
    __cudampi__spsaAdaptiveOpt.base_x[i]    = pnow;
    __cudampi__spsaAdaptiveOpt.m[i] = 0.0;
    __cudampi__spsaAdaptiveOpt.v[i] = 0.0;
    __cudampi__simpleAdaptiveOpt.base_x[i]  = pnow;
    __cudampi__simpleAdaptiveOpt.m[i] = 0.0;
    __cudampi__simpleAdaptiveOpt.v[i] = 0.0;
  }
}

void __cudampi__loadAndLogPowercapConfig(void) {
  powercap_config_t file_config;
  if (load_powercap_config("powercap.conf", &file_config) == 0) {
    __cudampi__powercapStrategy = file_config.strategy;
    if (file_config.strategy == CONTINUOUS_GREEDY || file_config.strategy == EQUAL_SHARE_CONTINUOUS_GREEDY) {
      __cudampi__cpu_min_powercap = file_config.cpu_min_powercap;
      __cudampi__gpu_min_powercap = file_config.gpu_min_powercap;
    } else if (file_config.strategy == EQUAL_SPLIT) {
      // EQUAL_SPLIT uses start_powercap per device; no global cap
      __cudampi__globalpowerlimit = 0;
      __cudampi__gradient_opt_start_powercap = file_config.start_powercap;
    } else if (file_config.strategy == EQUAL_SPLIT_EDP_MONITOR) {
      // Fixed equal caps like EQUAL_SPLIT, but the manager runs every window to
      // measure and log EDP samples (no optimization).
      __cudampi__globalpowerlimit = 0;
      __cudampi__gradient_opt_start_powercap = file_config.start_powercap;
      __cudampi__optimizer_step_interval = file_config.optimizer_step_interval > 0 ? file_config.optimizer_step_interval : 1ULL;
      __cudampi__cpu_min_powercap = file_config.cpu_min_powercap > 0.0f ? file_config.cpu_min_powercap : 0.10f;
      __cudampi__gpu_min_powercap = file_config.gpu_min_powercap > 0.0f ? file_config.gpu_min_powercap : 0.10f;
    } else if (file_config.strategy == EDP_MONITOR_SPLIT) {
      // Like EQUAL_SPLIT_EDP_MONITOR, but with independent GPU / CPU fixed caps.
      __cudampi__globalpowerlimit = 0;
      __cudampi__gradient_opt_start_powercap = file_config.start_powercap;
      __cudampi__gpu_powercap = file_config.gpu_powercap;
      __cudampi__cpu_powercap = file_config.cpu_powercap;
      __cudampi__optimizer_step_interval = file_config.optimizer_step_interval > 0 ? file_config.optimizer_step_interval : 1ULL;
      __cudampi__cpu_min_powercap = file_config.cpu_min_powercap > 0.0f ? file_config.cpu_min_powercap : 0.10f;
      __cudampi__gpu_min_powercap = file_config.gpu_min_powercap > 0.0f ? file_config.gpu_min_powercap : 0.10f;
    } else if (file_config.strategy == EDP_GRADIENT_SIMPLE || file_config.strategy == EDP_GRADIENT_SPSA || file_config.strategy == EDP_GRADIENT_CMAES) {
      __cudampi__globalpowerlimit = 0;
      __cudampi__gradient_opt_start_powercap = file_config.start_powercap;
      __cudampi__gradient_start_alpha = file_config.start_alpha;
      __cudampi__gradient_alpha_decay = file_config.alpha_decay;
      __cudampi__gradient_opt_eps = file_config.gradient_opt_eps;
      __cudampi__epsilon_decay = file_config.epsilon_decay;
      __cudampi__edp_optimization_steps = file_config.edp_optimization_steps;
      __cudampi__optimizer_step_interval = file_config.optimizer_step_interval > 0 ? file_config.optimizer_step_interval : 1ULL;
      __cudampi__cpu_min_powercap = file_config.cpu_min_powercap > 0.0f ? file_config.cpu_min_powercap : 0.10f;
      __cudampi__gpu_min_powercap = file_config.gpu_min_powercap > 0.0f ? file_config.gpu_min_powercap : 0.10f;
    } else if (file_config.strategy == EDP_SEARCH_SHARED || file_config.strategy == EDP_SEARCH_SPLIT) {
      __cudampi__globalpowerlimit = 0;
      __cudampi__gradient_opt_start_powercap = file_config.start_powercap;
      __cudampi__optimizer_step_interval = file_config.optimizer_step_interval > 0 ? file_config.optimizer_step_interval : 1ULL;
      __cudampi__cpu_min_powercap = file_config.cpu_min_powercap > 0.0f ? file_config.cpu_min_powercap : 0.10f;
      __cudampi__gpu_min_powercap = file_config.gpu_min_powercap > 0.0f ? file_config.gpu_min_powercap : 0.10f;
      __cudampi__search_depth = file_config.search_depth;
      __cudampi__search_grid_points = file_config.search_grid_points;
    }
    __cudampi__cpu_time_window_us = file_config.cpu_time_window_us;
    // Copy optional per-device cap override (used by EQUAL_SPLIT / EDP_MONITOR)
    __cudampi__deviceCapOverrideCount = file_config.device_powercaps_count;
    for (int i = 0; i < file_config.device_powercaps_count && i < __CUDAMPI_MAX_THREAD_COUNT; i++) {
      __cudampi__deviceCapOverride[i] = file_config.device_powercaps[i];
    }
    if (__cudampi__deviceCapOverrideCount > 0) {
      log_message(LOG_WARN, "Per-device cap override active for %d devices (device_powercaps from config)",
                  __cudampi__deviceCapOverrideCount);
    }
    // Apply global powercap from config if provided (> 0)
    if (__cudampi__powercapStrategy != DISABLED && file_config.global_powercap > 0.0f) {
      log_message(LOG_INFO, "Setting global powercap from config: %f", file_config.global_powercap);
      __cudampi__globalpowerlimit = file_config.global_powercap;
    }
  }
  else {
    __cudampi__powercapStrategy = DISABLED;
  }


  switch (__cudampi__powercapStrategy) {
    case DISABLED:
      log_message(LOG_INFO, "Powercap strategy: DISABLED");
      break;
    case BINARY_GREEDY:
      log_message(LOG_INFO, "Powercap strategy: BINARY_GREEDY");
      break;
    case EQUAL_SHARE_BINARY_GREEDY:
      log_message(LOG_INFO, "Powercap strategy: EQUAL_SHARE_BINARY_GREEDY");
      break;
    case CONTINUOUS_GREEDY:
      log_message(LOG_INFO, "Powercap strategy: CONTINUOUS_GREEDY");
      log_message(LOG_INFO, "CPU min powercap (part of range): %f", __cudampi__cpu_min_powercap);
      log_message(LOG_INFO, "GPU min powercap (part of range): %f", __cudampi__gpu_min_powercap);
      break;
    case EQUAL_SHARE_CONTINUOUS_GREEDY:
      log_message(LOG_INFO, "Powercap strategy: EQUAL_SHARE_CONTINUOUS_GREEDY");
      log_message(LOG_INFO, "CPU min powercap (part of range): %f", __cudampi__cpu_min_powercap);
      log_message(LOG_INFO, "GPU min powercap (part of range): %f", __cudampi__gpu_min_powercap);
      break;
    case EQUAL_SPLIT:
      log_message(LOG_INFO, "Powercap strategy: EQUAL_SPLIT");
      log_message(LOG_INFO, "start_powercap (part of range): %f", __cudampi__gradient_opt_start_powercap);
      break;
    case EQUAL_SPLIT_EDP_MONITOR:
      log_message(LOG_INFO, "Powercap strategy: EQUAL_SPLIT_EDP_MONITOR (fixed caps, EDP logging only)");
      log_message(LOG_INFO, "start_powercap (part of range): %f", __cudampi__gradient_opt_start_powercap);
      break;
    case EDP_GRADIENT_SIMPLE:
      log_message(LOG_INFO, "Powercap strategy: EDP_GRADIENT_SIMPLE");
      log_message(LOG_INFO, "Gradient params: start_alpha=%f, alpha_decay=%f, eps=%f", __cudampi__gradient_start_alpha, __cudampi__gradient_alpha_decay, __cudampi__gradient_opt_eps);
      log_message(LOG_INFO, "Dynamic lower bounds: cpu_min_powercap=%f gpu_min_powercap=%f", __cudampi__cpu_min_powercap, __cudampi__gpu_min_powercap);
      log_message(LOG_INFO, "EDP optimisation steps limit: %llu (0=unlimited)", __cudampi__edp_optimization_steps);
      break;
    case EDP_GRADIENT_SPSA:
      log_message(LOG_INFO, "Powercap strategy: EDP_GRADIENT_SPSA");
      log_message(LOG_INFO, "Gradient params: start_alpha=%f, alpha_decay=%f, eps=%f", __cudampi__gradient_start_alpha, __cudampi__gradient_alpha_decay, __cudampi__gradient_opt_eps);
      log_message(LOG_INFO, "Dynamic lower bounds: cpu_min_powercap=%f gpu_min_powercap=%f", __cudampi__cpu_min_powercap, __cudampi__gpu_min_powercap);
      log_message(LOG_INFO, "EDP optimisation steps limit: %llu (0=unlimited)", __cudampi__edp_optimization_steps);
      break;
    case EDP_GRADIENT_CMAES:
      log_message(LOG_INFO, "Powercap strategy: EDP_GRADIENT_CMAES (CMA-ES)");
      log_message(LOG_INFO, "CMA-ES params: start_powercap=%f, sigma0~eps=%f", __cudampi__gradient_opt_start_powercap, __cudampi__gradient_opt_eps);
      log_message(LOG_INFO, "Dynamic lower bounds: cpu_min_powercap=%f gpu_min_powercap=%f", __cudampi__cpu_min_powercap, __cudampi__gpu_min_powercap);
      log_message(LOG_INFO, "EDP optimisation steps limit: %llu (0=unlimited)", __cudampi__edp_optimization_steps);
      break;
    case EDP_MONITOR_SPLIT:
      log_message(LOG_INFO, "Powercap strategy: EDP_MONITOR_SPLIT (fixed independent GPU/CPU caps, EDP logging only)");
      log_message(LOG_INFO, "gpu_powercap (part of range): %f (-1=unset->start_powercap)", __cudampi__gpu_powercap);
      log_message(LOG_INFO, "cpu_powercap (part of range): %f (-1=unset->start_powercap)", __cudampi__cpu_powercap);
      log_message(LOG_INFO, "start_powercap fallback (part of range): %f", __cudampi__gradient_opt_start_powercap);
      break;
    case EDP_SEARCH_SHARED:
    case EDP_SEARCH_SPLIT:
      log_message(LOG_INFO, "Powercap strategy: %s", __cudampi__powercapStrategy == EDP_SEARCH_SPLIT ? "EDP_SEARCH_SPLIT (2D GPU/CPU grid)" : "EDP_SEARCH_SHARED (1D shared grid)");
      log_message(LOG_INFO, "Search params: depth=%d, grid_points=%d (full-range grid: spacing=1/gp, refine x2/(gp+1))",
                  __cudampi__search_depth, __cudampi__search_grid_points);
      log_message(LOG_INFO, "Dynamic lower bounds: cpu_min_powercap=%f gpu_min_powercap=%f", __cudampi__cpu_min_powercap, __cudampi__gpu_min_powercap);
      break;
    default:
      log_message(LOG_INFO, "Powercap strategy: UNKNOWN (%d)", __cudampi__powercapStrategy);
      break;
  }

  if (__cudampi__powercapStrategy != DISABLED) {
    log_message(LOG_INFO, "CPU power cap time window: %lld", __cudampi__cpu_time_window_us);
  }
}

void __cudampi__allocAndGatherPowercapRanges(void) {
  __cudampi__perNodePowerCapRange = (perNodePowerCapRange_t *)malloc(sizeof(perNodePowerCapRange_t) * __cudampi__MPIproccount);
  if (!__cudampi__perNodePowerCapRange) {
    log_message(LOG_ERROR,"\nNot enough memory");
    exit(-1);
  }

  // Broadcast CPU time window (us) so slaves use consistent value
  MPI_Bcast(&__cudampi__cpu_time_window_us, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Initialize power cap ranges for local node
  __cudampi__localPowerCapRange.cpuRange = __cudampi__getCpuPowerCapRange();
  for (int i = 0; i < __cudampi__localGpuDeviceCount; i++) {
    __cudampi__localPowerCapRange.gpuRange[i] = __cudampi__getGpuPowerCapRange(i);
  }

  MPI_Allgather(&__cudampi__localPowerCapRange, sizeof(perNodePowerCapRange_t), MPI_BYTE,
                __cudampi__perNodePowerCapRange, sizeof(perNodePowerCapRange_t), MPI_BYTE, MPI_COMM_WORLD);

  for (int i = 0; i < __cudampi__MPIproccount; i++) {
    log_message(LOG_DEBUG, "Node %d CPU Power Cap Range: %f - %f", i, __cudampi__perNodePowerCapRange[i].cpuRange.min, __cudampi__perNodePowerCapRange[i].cpuRange.max);
    for (int j = 0; j < __cudampi__GPUcountspernode[i]; j++) {
      log_message(LOG_DEBUG, "Node %d GPU %d Power Cap Range: %f - %f", i, j, __cudampi__perNodePowerCapRange[i].gpuRange[j].min, __cudampi__perNodePowerCapRange[i].gpuRange[j].max);
    }
  }
}

void __cudampi__initDevicePowercapConfig(void) {
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    __cudampi__devicePowerConfig[i].currentPower = -1; // initial value
    __cudampi__devicePowerConfig[i].deviceEnabled = 1;

    const int rank = __cudampi_targetMPIrankfordevice[i];
    if (i < __cudampi_totalgpudevicecount) {
      const int gpu = __cudampi_targetGPUfordevice[i];
      __cudampi__devicePowerConfig[i].powercapRange = __cudampi__perNodePowerCapRange[rank].gpuRange[gpu];
      __cudampi__devicePowerConfig[i].minPowerCap = getPowerCapFromRange(
        __cudampi__devicePowerConfig[i].powercapRange.min,
        __cudampi__devicePowerConfig[i].powercapRange.max,
        __cudampi__gpu_min_powercap);
    } else {
      __cudampi__devicePowerConfig[i].powercapRange = __cudampi__perNodePowerCapRange[rank].cpuRange;
      __cudampi__devicePowerConfig[i].minPowerCap = getPowerCapFromRange(
        __cudampi__devicePowerConfig[i].powercapRange.min,
        __cudampi__devicePowerConfig[i].powercapRange.max,
        __cudampi__cpu_min_powercap);
    }
  }

  // Emit the device->(type, MPI rank) mapping once so log-based analysis can
  // classify each device index as master GPU / slave GPU / slave CPU. Rank 0 is
  // the master node. cap-vector positions in [OPTIMIZER STEP] match these indices.
  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    const int rank = __cudampi_targetMPIrankfordevice[i];
    const int isGpu = (i < __cudampi_totalgpudevicecount);
    const char *type = isGpu ? "GPU" : "CPU";
    const char *role = (rank == 0) ? "master" : "slave";
    log_message(LOG_WARN, "[DEVICE TOPOLOGY] index=%d type=%s rank=%d role=%s min=%.3f max=%.3f",
                i, type, rank, role,
                __cudampi__devicePowerConfig[i].powercapRange.min,
                __cudampi__devicePowerConfig[i].powercapRange.max);
  }
}

static void __cudampi__initCmaes(void) {
  cmaesOpt_t* g = &__cudampi__cmaesOpt;
  g->n = __cudampi_totaldevicecount;
  g->terminated = 0;
  g->k = 0;
  g->mode = CMA_SAMPLE;

  int lam = 4 + (int)(3.0 * log((double)g->n + 1.0));
  if (lam > __CUDAMPI_CMAES_MAX_LAMBDA) lam = __CUDAMPI_CMAES_MAX_LAMBDA;
  g->lambda = lam;

  double* xstart = (double*)malloc(sizeof(double) * g->n);
  double* stddev = (double*)malloc(sizeof(double) * g->n);
  for (int i = 0; i < g->n; ++i) {
    g->phys_minv[i] = __cudampi__devicePowerConfig[i].powercapRange.min;
    g->phys_maxv[i] = __cudampi__devicePowerConfig[i].powercapRange.max;
    g->minv[i] = __cudampi__deviceLowerNorm(i);
    g->maxv[i] = 1.0;
    // Use user-provided epsilon directly in normalized space as CMA-ES initial stddev
    double eps_norm = (__cudampi__gradient_opt_eps > 0 ? __cudampi__gradient_opt_eps : 0.1);
    if (eps_norm > 0.5) eps_norm = 0.5;
    if (eps_norm < 1e-6) eps_norm = 1e-6;
    stddev[i] = eps_norm;
  }
  cmaes_boundary_transformation_init(&g->bounds, g->minv, g->maxv, g->n);
  for (int i = 0; i < g->n; ++i) {
    double cap = (double)__cudampi__devicePowerConfig[i].currentPowerCap;
    g->x_in_bounds[i] = __cudampi__clampNormForDevice(i, __cudampi__cap_to_norm(cap, g->phys_minv[i], g->phys_maxv[i]));
  }
  cmaes_boundary_transformation_inverse(&g->bounds, g->x_in_bounds, xstart, g->n);
  cmaes_init(&g->evo, g->n, xstart, stddev, 0, g->lambda, "cmaes_initials.par");
  (void)cmaes_SayHello(&g->evo);
  free(xstart);
  free(stddev);
}

static void __cudampi__initSearch(void) {
  searchOpt_t* g = &__cudampi__searchOpt;
  // SPLIT searches a GPU axis and a CPU axis (2D). With no CPU devices (GPU-only
  // run) the CPU axis has no effect, so collapse to a 1D search to avoid wasting
  // npoints x the evaluations on a redundant axis.
  int hasCpu = (__cudampi_totaldevicecount > __cudampi_totalgpudevicecount);
  g->dims = (__cudampi__powercapStrategy == EDP_SEARCH_SPLIT && hasCpu) ? 2 : 1;
  g->npoints = __cudampi__search_grid_points >= 2 ? __cudampi__search_grid_points : 2;
  g->depth = __cudampi__search_depth >= 1 ? __cudampi__search_depth : 1;
  // Level 0 covers the full normalized range [0,1], centred at 0.5 with spacing
  // 1/npoints, so the points are the npoints equal-bin midpoints (e.g. npoints=5
  // -> 0.1,0.3,0.5,0.7,0.9). Per-device lower bounds are applied via clamping.
  g->spacing = 1.0 / (double)g->npoints;
  g->center[0] = 0.5;
  g->center[1] = 0.5;
  g->level = 0;
  g->total = (g->dims == 2) ? (g->npoints * g->npoints) : g->npoints;
  g->k = 0;
  g->bestJ = 0.0;
  g->have_best = 0;
  g->terminated = 0;
  g->mode = SEARCH_EVAL;

  log_message(LOG_INFO, "SEARCH: init dims=%d npoints=%d depth=%d spacing0=%.4f (full-range grid, centre=0.5) total/level=%d",
              g->dims, g->npoints, g->depth, g->spacing, g->total);
  // Seed grid point 0 as the initial caps (field only; pushed later by
  // __cudampi__applyAllPowercaps once communicators exist) so the first optimiser
  // step measures point 0's EDP.
  __cudampi__searchInitCaps(g);
}

void __cudampi__applyInitialPowercapsForStrategy(void) {
  if (__cudampi__powercapStrategy == DISABLED) {
    return;
  }

  if(__cudampi__powercapStrategy == BINARY_GREEDY || __cudampi__powercapStrategy == EQUAL_SHARE_BINARY_GREEDY) {
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      __cudampi__devicePowerConfig[i].currentPowerCap = __cudampi__devicePowerConfig[i].powercapRange.defaultPowerCap;
    }
  }

  if (
      __cudampi__powercapStrategy == EQUAL_SPLIT ||
      __cudampi__powercapStrategy == EQUAL_SPLIT_EDP_MONITOR) {
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      float startPowerCap = __cudampi__gradient_opt_start_powercap;
      float lowerPowerCap = (float)__cudampi__deviceLowerNorm(i);
      if (startPowerCap < lowerPowerCap) {
        startPowerCap = lowerPowerCap;
      }
      __cudampi__devicePowerConfig[i].currentPowerCap = getPowerCapFromRange(
        __cudampi__devicePowerConfig[i].powercapRange.min,
        __cudampi__devicePowerConfig[i].powercapRange.max,
        startPowerCap);
    }

    // Optional per-device override: replace the uniform cap for any device that
    // has an explicit fraction in device_powercaps (clamped to its lower bound).
    // Lets the SNR collector hold all devices at base_cap while bumping one by eps.
    if (__cudampi__deviceCapOverrideCount > 0) {
      for (int i = 0; i < __cudampi_totaldevicecount && i < __cudampi__deviceCapOverrideCount; i++) {
        float frac = __cudampi__deviceCapOverride[i];
        float lowerPowerCap = (float)__cudampi__deviceLowerNorm(i);
        if (frac < lowerPowerCap) {
          frac = lowerPowerCap;
        }
        if (frac > 1.0f) {
          frac = 1.0f;
        }
        __cudampi__devicePowerConfig[i].currentPowerCap = getPowerCapFromRange(
          __cudampi__devicePowerConfig[i].powercapRange.min,
          __cudampi__devicePowerConfig[i].powercapRange.max,
          frac);
        log_message(LOG_WARN, "[CAP OVERRIDE] device %d set to fraction %.4f -> %.3f W",
                    i, frac, __cudampi__devicePowerConfig[i].currentPowerCap);
      }
    }
  }

  if (__cudampi__powercapStrategy == EDP_MONITOR_SPLIT) {
    // Independent fixed caps: every GPU device to gpu_powercap, every CPU device to
    // cpu_powercap. An unset fraction (< 0) falls back to the uniform start cap.
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      int isGPU = (i < __cudampi_totalgpudevicecount);
      float frac = isGPU ? __cudampi__gpu_powercap : __cudampi__cpu_powercap;
      if (frac < 0.0f) {
        frac = __cudampi__gradient_opt_start_powercap;
      }
      float lowerPowerCap = (float)__cudampi__deviceLowerNorm(i);
      if (frac < lowerPowerCap) {
        frac = lowerPowerCap;
      }
      if (frac > 1.0f) {
        frac = 1.0f;
      }
      __cudampi__devicePowerConfig[i].currentPowerCap = getPowerCapFromRange(
        __cudampi__devicePowerConfig[i].powercapRange.min,
        __cudampi__devicePowerConfig[i].powercapRange.max,
        frac);
      log_message(LOG_INFO, "[MONITOR SPLIT] %s device %d set to fraction %.4f -> %.3f W",
                  isGPU ? "GPU" : "CPU", i, frac, __cudampi__devicePowerConfig[i].currentPowerCap);
    }
  }

  if (
      __cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE ||
      __cudampi__powercapStrategy == EDP_GRADIENT_SPSA ||
      __cudampi__powercapStrategy == EDP_GRADIENT_CMAES) {
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      __cudampi__devicePowerConfig[i].currentPowerCap = getPowerCapFromRange(
        __cudampi__devicePowerConfig[i].powercapRange.min,
        __cudampi__devicePowerConfig[i].powercapRange.max,
        __cudampi__gradient_opt_start_powercap);
    }
    __cudampi__initializeGradientOpt(__cudampi__gradient_start_alpha, __cudampi__gradient_opt_eps);
    if (__cudampi__powercapStrategy == EDP_GRADIENT_CMAES) {
      __cudampi__initCmaes();
    }
  }

  if (__cudampi__powercapStrategy == EDP_SEARCH_SHARED ||
      __cudampi__powercapStrategy == EDP_SEARCH_SPLIT) {
    // initSearch sets currentPowerCap for every device to the first grid point.
    __cudampi__initSearch();
  }

  if ((__cudampi__powercapStrategy == CONTINUOUS_GREEDY || __cudampi__powercapStrategy == EQUAL_SHARE_CONTINUOUS_GREEDY) && __cudampi__globalpowerlimit > 0.0f) {
    float totalMinPowerCap = 0.0f;
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      totalMinPowerCap += __cudampi__devicePowerConfig[i].minPowerCap;
    }
    log_message(LOG_INFO, "Total min power cap: %f", totalMinPowerCap);

    if (totalMinPowerCap >= __cudampi__globalpowerlimit) {
      log_message(LOG_INFO, "Setting all devices to minimum power cap.");
      for (int i = 0; i < __cudampi_totaldevicecount; i++) {
        __cudampi__devicePowerConfig[i].currentPowerCap = __cudampi__devicePowerConfig[i].minPowerCap;
      }
    } else {
      log_message(LOG_INFO, "Distributing extra power proportionally among devices.");
      float extraPower = __cudampi__globalpowerlimit - totalMinPowerCap;
      log_message(LOG_INFO, "Extra power: %f", extraPower);

      float totalFreeCapacity = 0.0f;
      for (int i = 0; i < __cudampi_totaldevicecount; i++) {
        __cudampi__devicePowerConfig[i].currentPowerCap = __cudampi__devicePowerConfig[i].minPowerCap;
        totalFreeCapacity += (__cudampi__devicePowerConfig[i].powercapRange.max - __cudampi__devicePowerConfig[i].minPowerCap);
      }

      if (extraPower >= totalFreeCapacity) {
        log_message(LOG_INFO, "Extra power is more than or equal to total available free capacity. Setting all to max.");
        for (int i = 0; i < __cudampi_totaldevicecount; i++) {
          __cudampi__devicePowerConfig[i].currentPowerCap = __cudampi__devicePowerConfig[i].powercapRange.max;
        }
      } else if (totalFreeCapacity > 0.0f) {
        for (int i = 0; i < __cudampi_totaldevicecount; i++) {
          float freeCapacity = __cudampi__devicePowerConfig[i].powercapRange.max - __cudampi__devicePowerConfig[i].minPowerCap;
          __cudampi__devicePowerConfig[i].currentPowerCap += extraPower * (freeCapacity / totalFreeCapacity);
        }
      }
    }
  }

  for (int i = 0; i < __cudampi_totaldevicecount; i++) {
    log_message(LOG_INFO, "Power capping configuration for device %d:", i);
    log_message(LOG_INFO, "  Enabled         : %d", __cudampi__devicePowerConfig[i].deviceEnabled);
    log_message(LOG_INFO, "  Current Power Cap   : %f", __cudampi__devicePowerConfig[i].currentPowerCap);
    log_message(LOG_INFO, "  Minimum Power Cap   : %f", __cudampi__devicePowerConfig[i].minPowerCap);
    log_message(LOG_INFO, "  Power Cap Range : [%f, %f]",
         __cudampi__devicePowerConfig[i].powercapRange.min,
         __cudampi__devicePowerConfig[i].powercapRange.max);
  }
}

void __cudampi__powercappingManagerStep(void) {
  int amimanager;
  double combinedPower = 0.0;
  double combinedBatches = 0.0; // fractional: truncating to an integer quantizes EDP badly when a window holds few batches
  double sumTimePerDataPoint = 0.0;
  double sumRecipTimePerDataPoint = 0.0; // sum over devices of 1 / t_dp

  omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  amimanager = __cudampi__amimanager[__cudampi__currentDevice];
  omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

  static int selecteddevices = 0; // only updated by manager thread

  if (!amimanager || __cudampi__powercapStrategy == DISABLED) return;

  static unsigned long long managerExecutionCount = 0ULL;
  unsigned long long currentManagerExecution = 0ULL;
  #pragma omp atomic capture
  currentManagerExecution = ++managerExecutionCount;
  log_message(LOG_INFO, "Power capping manager execution #%llu on device %d with strategy %d",
              currentManagerExecution, __cudampi__currentDevice, __cudampi__powercapStrategy);

  if (__cudampi__powercapStrategy == BINARY_GREEDY || __cudampi__powercapStrategy == EQUAL_SHARE_BINARY_GREEDY) {
    if (!selecteddevices) {
      if (__cudampi__powercapStrategy == EQUAL_SHARE_BINARY_GREEDY) {
        selecteddevices = __cudampi__selectDevicesForPowerlimitGreedyEqualShare();
      } else {
        selecteddevices = __cudampi__selectdevicesforpowerlimit_greedy();
      }
    } else {
      float power = __cudampi__gettotalpowerofselecteddevices();
      if (power != (-1)) {
        if (power > __cudampi__globalpowerlimit) {
          log_message(LOG_DEBUG,"\ntotal power=%f limit=%f, adjusting", power, __cudampi__globalpowerlimit);
          fflush(stdout);
          if (__cudampi__powercapStrategy == EQUAL_SHARE_BINARY_GREEDY) {
            __cudampi__selectDevicesForPowerlimitGreedyEqualShare();
          } else {
            __cudampi__selectdevicesforpowerlimit_greedy();
          }
        }
      }
    }
  }
  else if (__cudampi__powercapStrategy == CONTINUOUS_GREEDY || __cudampi__powercapStrategy == EQUAL_SHARE_CONTINUOUS_GREEDY) {
    if (!selecteddevices) {
      if (__cudampi__powercapStrategy == EQUAL_SHARE_CONTINUOUS_GREEDY) {
        selecteddevices = __cudampi__SelectPowercapEqualEqualShare();
      } else {
        selecteddevices = __cudampi__selectpowercap_equal();
      }
    }
  }
  else if (
      __cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE ||
      __cudampi__powercapStrategy == EDP_GRADIENT_SPSA ||
      __cudampi__powercapStrategy == EDP_GRADIENT_CMAES ||
      __cudampi__powercapStrategy == EDP_SEARCH_SHARED ||
      __cudampi__powercapStrategy == EDP_SEARCH_SPLIT ||
      __cudampi__powercapStrategy == EQUAL_SPLIT_EDP_MONITOR ||
      __cudampi__powercapStrategy == EDP_MONITOR_SPLIT) {
    static unsigned long long edp_opt_iterations = 0ULL; // number of completed optimisation updates
    static int edp_opt_limit_logged = 0;                 // avoid spamming logs when limit reached
    static unsigned long long edp_window_count = 0ULL;   // total sync windows seen
    static unsigned long long acc_count = 0ULL;          // windows accumulated since last optimizer step
    static double acc_edp = 0.0;                         // sum of per-window EDP values
    // Windows still to discard before EDP is trusted again. A window that follows a cap
    // change mixes batches run under the old and the new caps; the first window of the
    // run contains start-up. Starts at 1 so that first window is dropped.
    static unsigned long long settle_windows_left = 1ULL;
    double combinedEnergy = 0.0;
    // Per-device energy/time accumulators as read in this step; subtracted (not zeroed)
    // at the end so energy reported by a device while the optimizer runs is kept.
    double consumedEnergy[__CUDAMPI_MAX_THREAD_COUNT];
    double consumedEnergyTime[__CUDAMPI_MAX_THREAD_COUNT];
    double consumedDataPoints[__CUDAMPI_MAX_THREAD_COUNT];
    double consumedDataTime[__CUDAMPI_MAX_THREAD_COUNT];

    // TODO: Reintroduce stricter EDP sample validation/rejection once this path has settled.
    // log_message(LOG_INFO, "EDP Gradient Optimization: Checking if all devices have completed their last batch.");
    int allDevicesCompleted = 1;
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_set_lock(&(__cudampi__devicelocks[i]));
      if (__cudampi__mgr_batches_sent[i] >= __cudampi__last_batches_sent[i]) {
        log_message(LOG_DEBUG, "Device %d has not completed its last batch yet. Current: %ld, Last: %ld", i, __cudampi__mgr_batches_sent[i], __cudampi__last_batches_sent[i]);
        allDevicesCompleted = 0;
        omp_unset_lock(&(__cudampi__devicelocks[i]));
        break;
      }
      // Device power = measured energy over all batches finished since the last window,
      // divided by the time those batches took. Devices synchronize asynchronously, so
      // that time differs from the manager's window length; averaging to power first
      // and multiplying by period_sec below charges every device for the same interval.
      consumedEnergy[i] = __cudampi__windowEnergy[i];
      consumedEnergyTime[i] = __cudampi__windowEnergyTime[i];
      if (consumedEnergyTime[i] > 0.0) {
        combinedPower += consumedEnergy[i] / consumedEnergyTime[i];
      } else {
        // No energy sample in this window (measurement failed): fall back to last known power
        combinedPower += __cudampi__devicePowerConfig[i].currentPower;
      }

      // Device throughput = data points of all batches finished since the last window,
      // divided by the time those batches took (same reasoning as for power above).
      consumedDataPoints[i] = __cudampi__windowDataPoints[i];
      consumedDataTime[i] = __cudampi__windowDataTime[i];
      if (consumedDataPoints[i] > 0.0 && consumedDataTime[i] > 0.0) {
        double timePerDataPoint = consumedDataTime[i] / consumedDataPoints[i];
        sumTimePerDataPoint += timePerDataPoint;
        sumRecipTimePerDataPoint += 1.0 / timePerDataPoint;
      } else if (__cudampi__timePerDataPoint[i] > 0.0 && isfinite(__cudampi__timePerDataPoint[i])) {
        // Nothing accumulated in this window: fall back to the last batch's rate
        sumTimePerDataPoint += __cudampi__timePerDataPoint[i];
        sumRecipTimePerDataPoint += 1.0 / __cudampi__timePerDataPoint[i];
      }
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }

    if(!allDevicesCompleted) {
      // log_message(LOG_INFO, "Not all devices have completed their last batch. Skipping optimization.");
      return;
    }
    // Measure time between optimizer syncs. For the first sync, use app start time
    struct timeval now;
    gettimeofday(&now, NULL);
    double period_sec = 0.0;
    if (!first_sync_done) {
      if (__cudampi__appStartTimestampSet) {
        period_sec = (double)(now.tv_sec - __cudampi__appStartTime.tv_sec)
                    + (double)(now.tv_usec - __cudampi__appStartTime.tv_usec) / 1000000.0;
        first_sync_done = 1;
      } else {
        log_message(LOG_ERROR, "App start time not set, cannot measure first optimizer sync period.");
        period_sec = 0.0;
      }
    } else {
      period_sec = (double)(now.tv_sec - prev_sync_time.tv_sec)
                  + (double)(now.tv_usec - prev_sync_time.tv_usec) / 1000000.0;
    }
    __cudampi__optimizerSyncPeriod = period_sec;
    prev_sync_time = now;

    // calculate edp per data point, but scale it by batch size to keep values in reasonable range
    // Estimate number of data points processed during this period using per-device t_per_dp
    // total_dp ~= period_sec * sum_i (1 / t_per_dp[i])
    double estimatedDataPointsD = period_sec * sumRecipTimePerDataPoint;
    if (estimatedDataPointsD < 0.0) estimatedDataPointsD = 0.0;
    if (__cudampi__default_batch_size > 0UL) {
      combinedBatches = estimatedDataPointsD / (double)__cudampi__default_batch_size;
    }
    if (!(combinedBatches > 0.0) || !isfinite(combinedBatches)) {
      log_message(LOG_WARN, "EDP estimated batch denominator is not positive (estimated_dp=%.3f). Using 1.", estimatedDataPointsD);
      combinedBatches = 1.0;
    }
    combinedEnergy = combinedPower * period_sec;

    if (combinedEnergy > 0.0 && period_sec > 0.0) {
      __cudampi__edp = (combinedEnergy * period_sec) /
                        (combinedBatches * combinedBatches);
    } else {
      __cudampi__edp = 0.0;
    }
    // A settling window is measured and consumed like any other, but its EDP is neither
    // logged as an [OPTIMIZER STEP] sample nor fed to the optimizer.
    int settleWindow = (settle_windows_left > 0ULL);
    if (settleWindow) {
      settle_windows_left--;
      log_message(LOG_WARN, "[OPTIMIZER SETTLE] discarded window period_s=%.6f edp=%.8f avg_power=%.3fW",
                  period_sec, __cudampi__edp, combinedPower);
    } else {
      edp_window_count++;
    }

    // Structured per-window EDP sample for SNR analysis
    if (!settleWindow) {
      char cap_str[1024] = "";
      int cap_pos = 0;
      for (int i = 0; i < __cudampi_totaldevicecount && cap_pos < (int)sizeof(cap_str) - 16; i++) {
        double minv = __cudampi__devicePowerConfig[i].powercapRange.min;
        double maxv = __cudampi__devicePowerConfig[i].powercapRange.max;
        double norm = (maxv > minv) ? (__cudampi__devicePowerConfig[i].currentPowerCap - minv) / (maxv - minv) : 0.0;
        cap_pos += snprintf(cap_str + cap_pos, sizeof(cap_str) - cap_pos, "%s%.4f", i ? "," : "", norm);
      }
      log_message(LOG_WARN, "[OPTIMIZER STEP] window=%llu interval=%llu phase=%d energy_J=%.3f period_s=%.6f batches=%.3f edp=%.8f avg_power=%.3fW cap=%s",
                  edp_window_count, __cudampi__optimizer_step_interval, __cudampi__powercapStrategy,
                  combinedEnergy, period_sec, combinedBatches, __cudampi__edp,
                  combinedPower, cap_str);
    }

    // Accumulate EDP for N-window averaging before optimizer step
    if (!settleWindow) {
      acc_edp += __cudampi__edp;
      acc_count++;
    }

    // Only trigger optimizer every optimizer_step_interval windows
    if (!settleWindow && acc_count >= __cudampi__optimizer_step_interval) {
      // Use mean EDP across accumulated windows as the optimizer input
      __cudampi__edp = acc_edp / (double)acc_count;
      acc_edp = 0.0; acc_count = 0ULL;
      log_message(LOG_WARN, "[EDP] edp=%.8f", __cudampi__edp);

      // If a limit is configured and already reached, stop further optimisation updates
      if (__cudampi__edp_optimization_steps > 0ULL && edp_opt_iterations >= __cudampi__edp_optimization_steps) {
        if (!edp_opt_limit_logged) {
          log_message(LOG_INFO, "EDP optimisation step limit reached (%llu). Skipping further optimisation.", __cudampi__edp_optimization_steps);
          edp_opt_limit_logged = 1;
        }
        // Mark optimisation finished and snapshot global stats once
        if (!__cudampi__optimizationFinished) {
          #pragma omp critical
          {
            if (!__cudampi__optimizationFinished) {
              __cudampi__optimizationFinished = 1;
              // Time since app start
              struct timeval finish_time;
              gettimeofday(&finish_time, NULL);
              if (__cudampi__appStartTimestampSet) {
                double finish_sec = (double)(finish_time.tv_sec - __cudampi__appStartTime.tv_sec)
                                  + (double)(finish_time.tv_usec - __cudampi__appStartTime.tv_usec) / 1000000.0;
                __cudampi__optimizationFinishedTime = finish_sec;
              } else {
                __cudampi__optimizationFinishedTime = 0.0;
              }
              // Energy snapshot
              __cudampi__optimizationFinishedEnergy = __cudampi__totalEnergyUsed;
              // Data points snapshot (sum over devices safely)
              unsigned long long total_dp = 0ULL;
              for (int i = 0; i < __cudampi_totaldevicecount; i++) {
                omp_set_lock(&(__cudampi__devicelocks[i]));
                total_dp += __cudampi__data_points_sent[i];
                omp_unset_lock(&(__cudampi__devicelocks[i]));
              }
              __cudampi__optimizationFinishedDataPoints = total_dp;
              log_message(LOG_INFO, "Dynamic optimisation finished: t=%.3fs, E=%.3fJ, DP=%llu", __cudampi__optimizationFinishedTime, __cudampi__optimizationFinishedEnergy, __cudampi__optimizationFinishedDataPoints);
            }
          }
        }
        // Do not call optimiser anymore, but keep counters in sync below
      } else {
        // Proceed with one optimisation update
        __cudampi__grad_update_performed = 0;
        float capsBefore[__CUDAMPI_MAX_THREAD_COUNT];
        for (int i = 0; i < __cudampi_totaldevicecount; i++) {
          capsBefore[i] = __cudampi__devicePowerConfig[i].currentPowerCap;
        }
        __cudampi__selectpowercap_gradient();
        // If any cap moved, the next window still contains work done under the old caps
        for (int i = 0; i < __cudampi_totaldevicecount; i++) {
          if (__cudampi__devicePowerConfig[i].currentPowerCap != capsBefore[i]) {
            settle_windows_left = 1ULL;
            break;
          }
        }
        if (__cudampi__grad_update_performed) {
          edp_opt_iterations++;
          __cudampi__grad_update_performed = 0;
        }
      }
    }

    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_set_lock(&(__cudampi__devicelocks[i]));
      __cudampi__mgr_batches_sent[i] = __cudampi__last_batches_sent[i];
      __cudampi__mgr_data_points_sent[i] = __cudampi__last_data_points_sent[i];
      __cudampi__windowEnergy[i] -= consumedEnergy[i];
      __cudampi__windowEnergyTime[i] -= consumedEnergyTime[i];
      __cudampi__windowDataPoints[i] -= consumedDataPoints[i];
      __cudampi__windowDataTime[i] -= consumedDataTime[i];
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }
  } 
}
