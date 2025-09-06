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
#include "powercap_config.h"
#include "cudampi_state.h"

powercapStrategy_t __cudampi__powercapStrategy = DISABLED;
perNodePowerCapRange_t* __cudampi__perNodePowerCapRange;
perNodePowerCapRange_t __cudampi__localPowerCapRange;
simpleGradientOpt_t __cudampi__simpleGradientOpt;
spsaGradientOpt_t __cudampi__spsaGradientOpt;

// Values below are expressed in terms of possible power cap range
// i.e. if min possible power cap is 100W and max is 250W, then 0.25 means 100W + 0.25 * (250W - 100W) = 137.5W
float __cudampi__cpu_min_powercap = 0.0;
float __cudampi__gpu_min_powercap = 0.0;
float __cudampi__gradient_opt_start_powercap = 1.0;

// CPU power cap time window (microseconds), broadcast to slaves
unsigned long long __cudampi__cpu_time_window_us = 1000000ULL; // default 1s

// Gradient optimisation runtime parameters (configurable via powercap.conf)
float __cudampi__gradient_start_alpha = 2.0f;
float __cudampi__gradient_alpha_decay = 0.99f;
float __cudampi__gradient_opt_eps = 5.0f;

double __cudampi__mgr_edp[__CUDAMPI_MAX_THREAD_COUNT];

float getPowerCapFromRange(float min, float max, float target) {
  // if min = 100W and max = 250W, then target = 0.25 means  this function should return 100W + 0.25 * (250W - 100W) = 137.5W
  return min + target * (max - min);
}

float getFreePowerCap(int index) {
  return __cudampi__devicePowerConfig[index].powercapRange.max - __cudampi__devicePowerConfig[index].currentPowerCap;
}

void setDevicePowerCap(int index) {
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

void __cudampi__updatePowerCap(float v, int i) {
  if (v < __cudampi__devicePowerConfig[i].powercapRange.min) {
    v = __cudampi__devicePowerConfig[i].powercapRange.min;
  } 
  else if (v > __cudampi__devicePowerConfig[i].powercapRange.max) {
    v = __cudampi__devicePowerConfig[i].powercapRange.max;
  }

  __cudampi__devicePowerConfig[i].currentPowerCap = v;
  setDevicePowerCap(i);
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

  switch (g->mode) {
    case PROBE_PLUS:
      log_message(LOG_INFO, "Gradient optimisation: entering PLUS_PROBE phase");

      for (int i = 0; i < n; ++i) {
        g->base_x[i] = (double)__cudampi__devicePowerConfig[i].currentPowerCap;

        __cudampi__generatePerturbation(g, i);
        double v = g->base_x[i] + (g->eps * g->delta[i]);

        log_message(LOG_INFO, "PROBE_PLUS: probe [%d] %f -> %f", i, __cudampi__devicePowerConfig[i].currentPowerCap, v);
        __cudampi__updatePowerCap((float)v, i);
      }
      
      g->mode = PROBE_MINUS;
      return;
    case PROBE_MINUS:
      log_message(LOG_INFO, "Gradient optimisation: entering PROBE_MINUS phase");

      g->J_plus = 0.0;
      for (int i = 0; i < n; ++i) {
        g->J_plus += __cudampi__mgr_edp[i];

        double v = g->base_x[i] - (g->eps * g->delta[i]);

        log_message(LOG_INFO, "PROBE_MINUS: probe [%d] %f -> %f", i, __cudampi__devicePowerConfig[i].currentPowerCap, v);
        __cudampi__updatePowerCap((float)v, i);
      }
      
      g->mode = DESCENT;
      return;
    case DESCENT:
      log_message(LOG_INFO, "Gradient optimisation: entering DESCENT phase");

      double J_minus = 0.0;
      double grad = 0.0;
      for (int i = 0; i < n; ++i) {
        J_minus += __cudampi__mgr_edp[i];
      }
      log_message(LOG_INFO, "DESCENT: J_plus=%lf J_minus=%lf", g->J_plus, J_minus);
      for (int i = 0; i < n; ++i) {
        grad = (g->J_plus - J_minus) / (2.0 * g->eps * g->delta[i]);
        
        double v = g->base_x[i] - (g->alpha * grad);

        log_message(LOG_INFO, "DESCENT: probe [%d] %f -> %f with grad=%lf", i, __cudampi__devicePowerConfig[i].currentPowerCap, v, grad);
        __cudampi__updatePowerCap((float)v, i);
      }
      
      g->alpha *= __cudampi__gradient_alpha_decay;
      g->mode = PROBE_PLUS;
      return;
    default:
      log_message(LOG_ERROR, "Gradient optimisation: invalid mode %d", g->mode);
      return;
  }
}
  
/*
 *
 * Single iteration of the forward-difference gradient-descent optimiser.
 * Called by the manager thread every time it has seen at least one new
 * batch from every device.  The routine owns no dynamic memory; all
 * optimiser state lives in the global simpleGradientOpt_t structure.
 *
 * Global inputs
 *   __cudampi_totaldevicecount              // dimension n of the vectors
 *   __cudampi__devicePowerConfig[i]         // current power cap (x_i)
 *   __cudampi__mgr_edp[i]                   // measured EDP (y_i)
 *
 * Global output
 *   __cudampi__devicePowerConfig[i].currentPowerCap
 *     // overwritten with either a probe cap or a descent cap
 *
 * Internal state fields (g points to __cudampi__gradientOpt)
 *   mode        0 = need base sample, 1 = collecting probes
 *   probe_dim   index of the coordinate currently being probed
 *   base_x[]    power caps captured at the base sample
 *   base_y[]    EDP vector captured at the base sample
 *   grad[]      gradient being assembled one component at a time
 *   eps         finite-difference step size
 *   alpha       learning rate for the descent step
 *
 * State-machine overview
 *   BASE phase   (mode == 0)
 *     - snapshot x and y into base_x / base_y
 *     - set probe_dim = 0
 *     - write first probe vector x + eps*e0 to the caps
 *     - switch to mode = 1 and return
 *
 *   PROBE phase  (mode == 1)
 *     - compute the gradient component for coordinate probe_dim
 *     - if probe_dim < n-1
 *         schedule the next probe (increment probe_dim) and return
 *       else
 *         compute the full descent step: x_next = base_x - alpha*grad
 *         clamp each element to legal limits via __cudampi__updatePowerCap
 *         write x_next to the caps and reset mode to 0
 */
void __cudampi__gradientOptStepSimple() {
  const int n = __cudampi_totaldevicecount;
  simpleGradientOpt_t* g = &__cudampi__simpleGradientOpt;

  // BASE phase: capture reference sample and emit first probe
  if (g->mode == BASE) {
    log_message(LOG_INFO, "Gradient optimisation: entering BASE phase");
    for (int i = 0; i < n; ++i) {
      g->base_x[i] = (double)__cudampi__devicePowerConfig[i].currentPowerCap;
      g->base_y[i] = __cudampi__mgr_edp[i];
    }
    g->probe_dim = 0;

    // first probe perturbs coordinate 0 by +eps
    for (int i = 0; i < n; ++i) {
      double v = g->base_x[i] + (i == g->probe_dim ? g->eps : 0.0);
      log_message(LOG_INFO, "Gradient optimisation: probe [%d] %f -> %f", i, __cudampi__devicePowerConfig[i].currentPowerCap, v);
      __cudampi__updatePowerCap((float)v, i);  // helper clamps to limits
    }

    g->mode = PROBE;  // enter PROBE phase
    return;
  }

  // PROBE phase: process result of a single coordinate probe
  double J_probe = 0.0;
  double J_base  = 0.0;
  for (int i = 0; i < n; ++i) {
    J_probe += __cudampi__mgr_edp[i];
    J_base  += g->base_y[i];
  }

  log_message(LOG_INFO, "Gradient optimisation: entering PROBE phase, J_probe = %f, J_base = %f", J_probe, J_base);

  // finite-difference estimate for current coordinate
  g->grad[g->probe_dim] = (J_probe - J_base) / g->eps;
  log_message(LOG_INFO, "Gradient optimisation: gradient [%d] = %f", g->probe_dim, g->grad[g->probe_dim]);
  ++g->probe_dim;

  // if more coordinates remain, schedule the next probe
  if (g->probe_dim < n) {
    for (int i = 0; i < n; ++i) {
      double v = g->base_x[i] + (i == g->probe_dim ? g->eps : 0.0);
      log_message(LOG_INFO, "Gradient optimisation: probe [%d] %f -> %f", i, __cudampi__devicePowerConfig[i].currentPowerCap, v);
      __cudampi__updatePowerCap((float)v, i);
    }
    return;  // stay in PROBE phase
  }

  // full gradient available: take one descent step
  for (int i = 0; i < n; ++i) {
    double v = g->base_x[i] - g->alpha * g->grad[i];
    log_message(LOG_INFO, "Gradient optimisation: descent [%d] %f -> %f", i, __cudampi__devicePowerConfig[i].currentPowerCap, v);
    __cudampi__updatePowerCap((float)v, i);  // helper clamps to limits
  }

  g->mode = BASE;  // restart cycle with a new base sample next time
  g->alpha *= __cudampi__gradient_alpha_decay;
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

  // Mode is now selected via strategy
  if (__cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE) {
    __cudampi__gradientOptStepSimple();
  } else if (__cudampi__powercapStrategy == EDP_GRADIENT_SPSA) {
    __cudampi__gradientOptStepSpsa();
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

void __cudampi__initializeGradientOpt (double alpha, double eps) {
  // For simplicity initialize both optimisers, but only one will be used
  __cudampi__simpleGradientOpt.alpha = alpha;
  __cudampi__simpleGradientOpt.eps = eps;
  __cudampi__simpleGradientOpt.mode = BASE;
  __cudampi__spsaGradientOpt.alpha = alpha;
  __cudampi__spsaGradientOpt.eps = eps;
  __cudampi__spsaGradientOpt.mode = PROBE_PLUS;
  for (int i = 0; i < __cudampi_totaldevicecount; ++i) {
    __cudampi__simpleGradientOpt.base_x[i]  = __cudampi__devicePowerConfig[i].currentPowerCap;
    __cudampi__spsaGradientOpt.base_x[i]  = __cudampi__devicePowerConfig[i].currentPowerCap;
  }
}

void __cudampi__loadAndLogPowercapConfig(void) {
  powercap_config_t file_config;
  if (load_powercap_config("powercap.conf", &file_config) == 0) {
    __cudampi__powercapStrategy = file_config.strategy;
    if (file_config.strategy == CONTINOUS_EQUAL) {
      __cudampi__cpu_min_powercap = file_config.cpu_min_powercap;
      __cudampi__gpu_min_powercap = file_config.gpu_min_powercap;
    } else if (file_config.strategy == EDP_GRADIENT_SIMPLE || file_config.strategy == EDP_GRADIENT_SPSA) {
      __cudampi__globalpowerlimit = 0;
      __cudampi__gradient_opt_start_powercap = file_config.start_powercap;
      __cudampi__gradient_start_alpha = file_config.start_alpha;
      __cudampi__gradient_alpha_decay = file_config.alpha_decay;
      __cudampi__gradient_opt_eps = file_config.gradient_opt_eps;
    }
    __cudampi__cpu_time_window_us = file_config.cpu_time_window_us;
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
    case CONTINOUS_EQUAL:
      log_message(LOG_INFO, "Powercap strategy: CONTINOUS_EQUAL");
      log_message(LOG_INFO, "CPU min powercap (part of range): %f", __cudampi__cpu_min_powercap);
      log_message(LOG_INFO, "GPU min powercap (part of range): %f", __cudampi__gpu_min_powercap);
      break;
    case EDP_GRADIENT_SIMPLE:
      log_message(LOG_INFO, "Powercap strategy: EDP_GRADIENT_SIMPLE");
      log_message(LOG_INFO, "Gradient params: start_alpha=%f, alpha_decay=%f, eps=%f", __cudampi__gradient_start_alpha, __cudampi__gradient_alpha_decay, __cudampi__gradient_opt_eps);
      break;
    case EDP_GRADIENT_SPSA:
      log_message(LOG_INFO, "Powercap strategy: EDP_GRADIENT_SPSA");
      log_message(LOG_INFO, "Gradient params: start_alpha=%f, alpha_decay=%f, eps=%f", __cudampi__gradient_start_alpha, __cudampi__gradient_alpha_decay, __cudampi__gradient_opt_eps);
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
}

void __cudampi__applyInitialPowercapsForStrategy(void) {
  if (__cudampi__powercapStrategy == DISABLED) {
    return;
  }

  if (__cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE || __cudampi__powercapStrategy == EDP_GRADIENT_SPSA) {
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      __cudampi__devicePowerConfig[i].currentPowerCap = getPowerCapFromRange(
        __cudampi__devicePowerConfig[i].powercapRange.min,
        __cudampi__devicePowerConfig[i].powercapRange.max,
        __cudampi__gradient_opt_start_powercap);
    }
    __cudampi__initializeGradientOpt(__cudampi__gradient_start_alpha, __cudampi__gradient_opt_eps);
  }

  if (__cudampi__powercapStrategy == CONTINOUS_EQUAL && __cudampi__globalpowerlimit > 0.0f) {
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
  omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  amimanager = __cudampi__amimanager[__cudampi__currentDevice];
  omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

  static int selecteddevices = 0; // only updated by manager thread

  if (!amimanager || __cudampi__powercapStrategy == DISABLED) return;

  if (__cudampi__powercapStrategy == BINARY_GREEDY) {
    if (!selecteddevices) {
      selecteddevices = __cudampi__selectdevicesforpowerlimit_greedy();
    } else {
      float power = __cudampi__gettotalpowerofselecteddevices();
      if (power != (-1)) {
        if (power > __cudampi__globalpowerlimit) {
          log_message(LOG_DEBUG,"\ntotal power=%f limit=%f, adjusting", power, __cudampi__globalpowerlimit);
          fflush(stdout);
          __cudampi__selectdevicesforpowerlimit_greedy();
        }
      }
    }
  }
  else if (__cudampi__powercapStrategy == CONTINOUS_EQUAL) {
    if (!selecteddevices) {
      selecteddevices = __cudampi__selectpowercap_equal();
    }
  }
  else if (__cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE || __cudampi__powercapStrategy == EDP_GRADIENT_SPSA) {
    log_message(LOG_INFO, "EDP Gradient Optimization: Checking if all devices have completed their last batch.");
    int allDevicesCompleted = 1;
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      omp_set_lock(&(__cudampi__devicelocks[i]));
      if (__cudampi__mgr_batches_sent[i] >= __cudampi__last_batches_sent[i]) {
        log_message(LOG_INFO, "Device %d has not completed its last batch yet. Current: %ld, Last: %ld", i, __cudampi__mgr_batches_sent[i], __cudampi__last_batches_sent[i]);
        allDevicesCompleted = 0;
        omp_unset_lock(&(__cudampi__devicelocks[i]));
        break;
      }
      __cudampi__mgr_edp[i] = (__cudampi__time_us[i] / 1000000.0) * __cudampi__devicePowerConfig[i].currentPower;
      omp_unset_lock(&(__cudampi__devicelocks[i]));
    }

    if(allDevicesCompleted) {
      log_message(LOG_INFO, "All devices have completed their last batch. Proceeding with optimization.");
      __cudampi__selectpowercap_gradient();
      for (int i = 0; i < __cudampi_totaldevicecount; i++) {
        omp_set_lock(&(__cudampi__devicelocks[i]));
        __cudampi__mgr_batches_sent[i] == __cudampi__last_batches_sent[i];
        omp_unset_lock(&(__cudampi__devicelocks[i]));
      }
    } else {
      log_message(LOG_INFO, "Not all devices have completed their last batch. Skipping optimization.");
    }
  }
}

void __cudampi__applyAllPowercaps(void) {
  // Push configured power caps to devices based on selected strategy
  if (__cudampi__powercapStrategy == CONTINOUS_EQUAL ||
      __cudampi__powercapStrategy == EDP_GRADIENT_SIMPLE ||
      __cudampi__powercapStrategy == EDP_GRADIENT_SPSA) {
    for (int i = 0; i < __cudampi_totaldevicecount; i++) {
      if (__cudampi__devicePowerConfig[i].currentPowerCap != -1) {
        setDevicePowerCap(i);
      }
    }
  }
}
