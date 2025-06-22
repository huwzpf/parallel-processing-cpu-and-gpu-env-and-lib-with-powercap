/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
// idea: in most cases, input data can be uploaded to GPU's memory and
// consequently we only need to copy a pointer in kernel invocation
// in OpenCL we could hide any kernel invocation

#include <stdio.h>
#include "math.h"

#define ENABLE_LOGGING
#include "logger.h"
#include "vecmaxdiv_defines.h"

void appkernel(void *devPtr, int num_elements, int num_threads) 
{
  double *devPtra = (double *)(((void **)devPtr)[0]);
  double *devPtrb = (double *)(((void **)devPtr)[1]);
  double *devPtrc = (double *)(((void **)devPtr)[2]);

  #pragma omp parallel for num_threads(num_threads)
for (long my_index = 0 ; my_index < num_elements; my_index++)
{
    long i;
    long result = 1;
    long max = sqrt(devPtra[my_index]);
    long elem = devPtra[my_index];
    for (i = 2; i < max; i++) 
    {
        if (!(elem % i)) 
        {
            if (i >= result) 
            {
                result = i;
            }
        }
    }
    max = sqrt(devPtrb[my_index]);
    elem = devPtrb[my_index];
    for (i = 2; i < max; i++) 
    {
        if (!(elem % i)) 
        {
            if (i >= result) 
            {
                result = i;
            }
        }
    }
    devPtrc[my_index] = result;
}
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long /* id */) 
{
    unsigned long num_elements = batchSize;
    log_message(LOG_DEBUG, "Launichng CPU Kernel with %i elements and %i threads.", num_elements, num_threads);
    appkernel(devPtr, num_elements, num_threads);
}

