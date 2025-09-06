/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef CUDAMPI_COMMON_H
#define CUDAMPI_COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <string.h>
#include "cudampi.h"

typedef struct {
    float min;
    float max;
    float defaultPowerCap;
    unsigned long long defaultTimeWindowUs; // Used just for CPU, but for now, let's keep it for simplicity
} powercapRange_t;

typedef struct 
{
  powercapRange_t gpuRange[MAX_GPU_PER_NODE];
  powercapRange_t cpuRange;
} perNodePowerCapRange_t;

typedef struct
{
    powercapRange_t powercapRange;
    float currentPower;
    float currentPowerCap;
    float minPowerCap;
    int deviceEnabled;
} devicePowerConfig_t;

typedef enum {
    PROBE_PLUS,
    PROBE_MINUS,
    DESCENT
} spsaState_t;

typedef enum {
    BASE,
    PROBE
} simpleGradState_t;


typedef struct {
    double  eps;                       
    double  alpha;
    spsaState_t     mode;
    double  base_x   [__CUDAMPI_MAX_THREAD_COUNT];
    double  delta   [__CUDAMPI_MAX_THREAD_COUNT];
    double  J_plus;
} spsaGradientOpt_t;

typedef struct {
    /* config */
    double  eps;                            /* finite-difference step  */
    double  alpha;                          /* learning rate           */
    simpleGradState_t     mode;
    int     probe_dim;                      /* which coordinate probe  */ 
    double  base_x   [__CUDAMPI_MAX_THREAD_COUNT];
    double  base_y   [__CUDAMPI_MAX_THREAD_COUNT];
    double  grad     [__CUDAMPI_MAX_THREAD_COUNT];
} simpleGradientOpt_t;

float computeDevPerformance(double period_us);

float getGPUpower(int gpuid);

cudaError_t getCpuEnergyUsed(float* lastEnergyMeasured, float* energyUsed);

cudaError_t __cudampi__getCpuFreeThreads(int* count);

void initializeCpuEnergyMeasurement(int* isInitialCpuEnergyMeasured, omp_lock_t* cpuEnergyLock, float* cpuLastEnergyMeasured);

powercapRange_t __cudampi__getCpuPowerCapRange();

powercapRange_t __cudampi__getGpuPowerCapRange(int gpuid);

void __cudampi__setGpuPowerCap(int gpuid, float powerCap);

void __cudampi__setCpuPowerCap(float powerCap, unsigned long long timeWindowUs);

#endif // CUDAMPI_COMMON_H
